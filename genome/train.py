import itertools as it
import os
import pickle
from typing import List, Optional

import binary_layers
import matplotlib.pyplot as plt
import models
import numpy as np
import tensorboard_easy
import tqdm
from genome import env_wrappers
from matplotlib import animation

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
    search_dist: models.NetworkDistribution64,
    env: env_wrappers.BinaryWrapper,
    run_name: str,
    learning_rate: float,
    population_size: int,
    n_generations: Optional[int] = None,
    max_episode_steps: Optional[int] = None,
    use_natural_gradient: bool = True,
    shape_fitness: bool = False,
    checkpoint_every: Optional[int] = None,
    render_every: Optional[int] = None,
):
    """
    Train a binary network on an OpenAI gym environment.

    Parameters
    ----------
    search_dist : NetworkDistribution64
        The search distribution to train. Modified in place.
    env : BinaryWrapper
        A wrapped gym environment to train on.
    run_name : str
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
        If provided, save a gif every `render_every` generations.
    """
    output_dir = os.path.join(OUTPUT_DIR, env.spec.id, run_name)
    log_dir = os.path.join(output_dir, LOG_DIR)
    ckpt_dir = os.path.join(output_dir, CHECKPOINT_DIR)
    anim_dir = os.path.join(output_dir, ANIMATION_DIR)
    os.makedirs(log_dir)
    os.makedirs(ckpt_dir)
    os.makedirs(anim_dir)

    def episode(binary_net: binary_layers.BinaryNetwork64) -> float:
        obs = env.reset()
        episode_return = 0

        for step in it.count():
            action = binary_net.forward(obs)
            obs, reward, done, _ = env.step(action)
            episode_return += reward

            if done or (max_episode_steps and step >= max_episode_steps):
                break

        return episode_return

    # TODO: replace with tensorboard video summaries
    def animate(binary_net: binary_layers.BinaryNetwork64, label: int) -> None:
        obs = env.reset()
        frames: List[np.ndarray] = []

        for step in it.count():
            frames.append(env.render(mode="rgb_array"))
            action = binary_net.forward(obs)
            obs, reward, done, _ = env.step(action)

            if done or (max_episode_steps and step >= max_episode_steps):
                break

        save_frames_as_gif(frames, os.path.join(anim_dir, f"train_{label}.gif"))

    with tensorboard_easy.Logger(log_dir) as logger:
        for step in tqdm.tqdm(
            range(n_generations) if n_generations else it.count(),
            desc="Training progress",
            unit=" generations",
            dynamic_ncols=True,
        ):
            evals = step * population_size

            # Periodically evaluate the MAP network, save gifs, and write checkpoints
            map_network = search_dist.binarize_maximum_likelihood()
            map_return = episode(map_network)

            if render_every and step % render_every == 0:
                animate(map_network, evals)

            if checkpoint_every and step % checkpoint_every == 0:
                with open(os.path.join(ckpt_dir, f"ckpt_{evals}.pkl"), "wb+") as f:
                    pickle.dump(search_dist, f)

            # Sample and evaluate test points
            test_points = []
            returns = np.empty(population_size)
            for i in range(population_size):
                point = search_dist.sample()
                point_return = episode(point)
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

            # TODO: plot bias histograms
            for i, layer in enumerate(search_dist.hidden_layers):
                logger.log_histogram(
                    f"hidden_{i}_weight_logits", layer.weight_logits, step=evals
                )
            logger.log_histogram(
                f"output_weight_logits",
                search_dist.output_layer.weight_logits,
                step=evals,
            )


# TODO: move to a dedicated run script
if __name__ == "__main__":
    env = env_wrappers.CartPoleV1()
    search_dist = models.NetworkDistribution64(
        num_hidden_layers=env.action_dims,
        num_outputs=1,
        init_weight_logits_std=0,
        use_bias=False,
        fixed_bias_std=1.0,
        init_bias_std=1.0,
    )

    train(
        search_dist=search_dist,
        env=env,
        run_name="test-refactor",
        learning_rate=0.1,
        population_size=256,
        n_generations=100,
        max_episode_steps=None,
        use_natural_gradient=True,
        shape_fitness=False,
        checkpoint_every=None,
        render_every=None,
    )
