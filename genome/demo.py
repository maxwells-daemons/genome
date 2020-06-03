from typing import List

import click

from genome import train, environments, distributions
from genome.binary_networks import inference


@click.command()
@click.argument("run-name", type=str)
@click.option(
    "--task", "-t", type=click.Choice(["CartPole", "LunarLander"]), default="CartPole"
)
@click.option("--layer", "-l", type=int, multiple=True, default=[64])
@click.option("--population-size", "-ps", type=int, default=64)
@click.option("--learning-rate", "-lr", type=float, default=0.06)
@click.option(
    "--implementation", "-i", type=click.Choice(["debug", "cpu", "gpu"]), default="cpu"
)
@click.option("--render-every", type=int, default=20)
def demo(
    run_name: str,
    task: str,
    layer: List[int],
    population_size: int,
    learning_rate: float,
    implementation: str,
    render_every: int,
):
    if task == "CartPole":
        env = environments.CartPoleV1()
    elif task == "LunarLander":
        env = environments.LunarLanderV2()
    else:
        raise ValueError(f"unrecognized task: {task}")

    if implementation == "debug":
        strategy = inference.InferenceStrategy.DEBUG
    elif implementation == "cpu":
        strategy = inference.InferenceStrategy.CPU64
    elif implementation == "gpu":
        strategy = inference.InferenceStrategy.GPU
    else:
        raise ValueError(f"unrecognized inference strategy: {implementation}")

    obs = env.reset()
    input_dims = len(obs)
    layer_dims = [input_dims] + list(layer) + [env.action_dims]
    search_dist = distributions.NetworkDistribution(
        layer_dims, init_weight_logits_std=3, use_bias=False, fixed_bias_std=3.0
    )

    train.train(
        search_dist=search_dist,
        inference_strategy=strategy,
        env=env,
        run_name=run_name,
        learning_rate=learning_rate,
        population_size=population_size,
        n_generations=None,
        max_episode_steps=None,
        use_natural_gradient=True,
        shape_fitness=False,
        checkpoint_every=None,
        render_every=render_every,
        save_renders=False,
    )


if __name__ == "__main__":
    demo()
