"""
A collection of wrappers around OpenAI Gym environments to allow using them with
binary networks.

NOTE: these aren't really "wrappers", since they wrap only one environment.
This is to bake the necessary discretization and output encoding into the environment.
"""

from typing import Any, Tuple, Union
import binary_layers

import gym
import numpy as np


class BinaryWrapper(gym.Wrapper):
    """
    A wrapper around an environment making it usable with binary networks.

    Only usable as an interface.
    Children must implement `action_dims`, `binarize_observation`, and `get_action`.
    """

    action_dims: int
    raw_observation: bool

    def __init__(self, env):
        super(BinaryWrapper, self).__init__(env)

    def reset(self, **kwargs) -> np.ndarray:
        observation = self.env.reset(**kwargs)
        return self.binarize_observation(observation)

    def step(self, binary_action: np.ndarray) -> Tuple[np.ndarray, float, bool, Any]:
        action = self.unbinarize_action(binary_action)
        observation, reward, done, info = self.env.step(action)
        return self.binarize_observation(observation), reward, done, info

    def binarize_observation(
        self, observation: np.ndarray
    ) -> Union[np.ndarray, np.uint64]:
        """
        Convert an observation into a valid input to a binary network.

        Transforms `observation` into a binary 64-vector.

        Parameters
        ----------
        observation : np.ndarray
            Observation vector output by the environment.

        Returns
        -------
        np.ndarray[64, bool] (if raw_observation = False)
            An array of 64 bools, acting as input to the binary network.
        np.uint64_t (if raw_observation = True)
            A packed binary 64-vector, acting as raw input to the binary network.
        """
        raise NotImplementedError

    def get_action(self, discrete_action: np.ndarray) -> Any:
        """
        Convert a discrete vector into a valid action in the environment.

        Parameters
        ----------
        discrete_action : np.ndarray[action_dims, int]
            A vector of integers output by the binary network's output layer.

        Returns
        -------
        Any
            The action to take in the environment. Type depends on the environment.
        """
        raise NotImplementedError


class CartPoleV1(BinaryWrapper):
    action_dims = 1
    raw_observation = False

    _n_buckets = 16
    _cart_pos_space = np.linspace(-2.4, 2.4, _n_buckets - 1)
    _cart_vel_space = np.linspace(-10, 10, _n_buckets - 1)
    _pole_angle_space = np.linspace(-41.8, 41.8, _n_buckets - 1)
    _pole_vel_space = np.linspace(-5, 5, _n_buckets - 1)

    def __init__(self):
        env = gym.make("CartPole-v1")
        super(CartPoleV1, self).__init__(env)

    def binarize_observation(self, observation: np.ndarray) -> np.ndarray:
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        cart_pos_i = np.searchsorted(self._cart_pos_space, cart_pos, side="left")
        cart_vel_i = np.searchsorted(self._cart_vel_space, cart_vel, side="left")
        pole_angle_i = np.searchsorted(self._pole_angle_space, pole_angle, side="left")
        pole_vel_i = np.searchsorted(self._pole_vel_space, pole_vel, side="left")

        discretized = np.zeros(64, bool)
        discretized[cart_pos_i] = True
        discretized[cart_vel_i + self._n_buckets] = True
        discretized[pole_angle_i + 2 * self._n_buckets] = True
        discretized[pole_vel_i + 3 * self._n_buckets] = True

        return discretized

    def unbinarize_action(self, binary_action: np.ndarray) -> bool:
        return bool(binary_action[0] > 0)


class CartPoleV1Bits(BinaryWrapper):
    action_dims = 1
    raw_observation = True

    def __init__(self):
        env = gym.make("CartPole-v1")
        super(CartPoleV1Bits, self).__init__(env)

    def binarize_observation(self, observation: np.ndarray) -> np.uint64:
        observation = observation.astype(np.float16, copy=False).view(np.uint16)
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        return binary_layers.concat_16bit(cart_pos, cart_vel, pole_angle, pole_vel)

    def unbinarize_action(self, binary_action: np.ndarray) -> bool:
        return bool(binary_action[0] > 0)
