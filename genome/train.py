import itertools as it
import os
import pickle
import scipy.special
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import distributions
import numpy as np
import tensorboard_easy
import tqdm
from matplotlib import animation

from genome import environments
from genome.binary_networks import inference

OUTPUT_DIR = "./outputs"
LOG_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"
ANIMATION_DIR = "animations"


# Adapted from: https://stackoverflow.com/questions/5284646/
# rank-items-in-an-array-using-python-numpy-without-sorting-array-twice
def rank(arr: np.ndarray) -> np.ndarray:
    """
    Get the rank of each item in an array.

    For example, rank([5.1, 0.4, 3.6]) = [3, 1, 2].

    Parameters
    ----------
    arr : np.ndarray
        An array of elements to rank.

    Returns
    -------
    np.ndarray
        Each input element's rank, as an ndarray of ints.
    """
    temp = arr.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(arr))
    return ranks


def rank_transform(returns: np.ndarray) -> np.ndarray:
    """
    Apply fitness shaping to a batch of returns.

    Used to make learning invariant to order-preserving transformations of the returns.

    Parameters
    ----------
    returns : np.ndarray
        A batch of episode returns.

    Returns
    -------
    np.ndarray
        The returns, transformed to being evenly spaced in the range [-0.5, 0.5].
    """
    scale = len(returns) - 1
    return (rank(returns) / scale) - 0.5


# Adapted from: https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
def save_frames_as_gif(frames: List[np.ndarray], path: str) -> None:
    """
    Write a gif from a list of image frames.

    Parameters
    ----------
    frames : [np.ndarray]
        List of frames, each stored as an ndarray.
    path : str
        The path to write the gif to.
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path, writer="imagemagick", fps=60)


def train(
    search_dist: distributions.NetworkDistribution,
    inference_strategy: inference.InferenceStrategy,
    env: environments.BinaryWrapper,
    run_name: str,
    learning_rate: float,
    population_size: int,
    n_generations: Optional[int] = None,
    max_episode_steps: Optional[int] = None,
    use_natural_gradient: bool = True,
    shape_fitness: bool = False,
    checkpoint_every: Optional[int] = None,
    render_every: Optional[int] = None,
    save_renders: bool = False,
):
    """
    Train a binary network on an OpenAI gym environment.

    Parameters
    ----------
    search_dist : NetworkDistribution
        The search distribution to train. Modified in place.
    inference_strategy : InferenceStrategy
        Which strategy to use for efficient inference.
    env : BinaryWrapper
        A wrapped gym environment to train on.
    run_name : int
        A name for this training run. Used to create files.
    learning_rate : float
        The learning rate to use.
    population_size : int
        How many individuals to sample each generation.
    n_generations : Optional[int] (default: None)
        The maximum number of optimization steps to perform.
        If None, train until the process is manually ended.
    max_episode_steps : Optional[int] (default: None)
        The maximum number of steps to run each evaluation for.
        If None, evaluations last until the gym environment says it is done.
    use_natural_gradient : bool (default: True)
        If True, use the natural gradient instead of the plain search gradient.
    shape_fitness : bool (default: False)
        If True, apply fitness shaping to generation returns.
    checkpoint_every : Optional[int] (default: None)
        If provided, checkpoint every `checkpoint_every` generations.
    render_every : Optional[int] (default: None)
        If provided, visualize every `render_every` generations.
    save_renders : bool (default: False)
        If True, save rendered scenes as gifs.
    """
    output_dir = os.path.join(OUTPUT_DIR, env.spec.id, run_name)
    log_dir = os.path.join(output_dir, LOG_DIR)
    ckpt_dir = os.path.join(output_dir, CHECKPOINT_DIR)
    anim_dir = os.path.join(output_dir, ANIMATION_DIR)
    os.makedirs(log_dir)
    os.makedirs(ckpt_dir)
    os.makedirs(anim_dir)

    compiled_network = inference.CompiledNetwork(
        search_dist.layer_dims, inference_strategy
    )

    def episode() -> float:
        obs = env.reset()
        episode_return = 0

        for step in it.count():
            action = compiled_network.forward(obs)
            obs, reward, done, _ = env.step(action)
            episode_return += reward

            if done or (max_episode_steps and step >= max_episode_steps):
                break

        return episode_return

    # TODO: replace with tensorboard video summaries
    def animate(label: int) -> None:
        obs = env.reset()

        if save_renders:
            frames: List[np.ndarray] = []

        for step in it.count():
            frame = env.render(mode="rgb_array")
            action = compiled_network.forward(obs)
            obs, _, done, _ = env.step(action)

            if save_renders:
                frames.append(frame)

            if done or (max_episode_steps and step >= max_episode_steps):
                break

        if save_renders:
            save_frames_as_gif(frames, os.path.join(anim_dir, f"train_{label}.gif"))

    def log_layer(layer: distributions.LayerDistribution, name: str, step: int):
        probs = scipy.special.expit(layer.weight_logits)
        variances = probs * (1 - probs)
        logger.log_histogram(f"{name}/weight_logits", layer.weight_logits, step=step)
        logger.log_histogram(f"{name}/weight_probs", probs, step=step)
        logger.log_histogram(f"{name}/weight_vars", variances, step=step)

    with tensorboard_easy.Logger(log_dir) as logger:
        for step in tqdm.tqdm(
            range(n_generations) if n_generations else it.count(),
            desc="Training progress",
            unit=" generations",
            dynamic_ncols=True,
        ):
            evals = step * population_size

            # Periodically evaluate the MAP network, save gifs, and write checkpoints
            compiled_network.set_params(search_dist.binarize_maximum_likelihood())
            map_return = episode()

            if render_every and step % render_every == 0:
                animate(evals)

            if checkpoint_every and step % checkpoint_every == 0:
                with open(os.path.join(ckpt_dir, f"ckpt_{evals}.pkl"), "wb+") as f:
                    pickle.dump(search_dist, f)

            # Sample and evaluate test points
            test_points: List[inference.NetworkParams] = []
            returns = np.empty(population_size)
            for i in range(population_size):
                point = search_dist.sample()
                compiled_network.set_params(point)
                point_return = episode()
                test_points.append(point)
                returns[i] = point_return

            # Apply fitness shaping
            if shape_fitness:
                shaped_returns = rank_transform(returns)
            else:
                shaped_returns = returns

            # Take a step of SGD
            samples_and_returns = list(zip(test_points, shaped_returns))
            search_dist.step(learning_rate, samples_and_returns, use_natural_gradient)

            # End-of-iter logging
            logger.log_scalar("mean_return", returns.mean(), step=evals)
            logger.log_scalar("map_return", map_return, step=evals)
            logger.log_histogram("sample_returns", returns, step=evals)

            for i, layer in enumerate(search_dist.layers[:-1]):
                log_layer(layer, f"hidden_{i}", evals)
            log_layer(search_dist.layers[-1], "output", evals)
