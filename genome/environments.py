"""
A collection of wrappers around OpenAI Gym environments to allow using them with
binary networks.

NOTE: these aren't really "wrappers", since they wrap only one environment.
This is to bake the necessary discretization and output encoding into the environment.
"""

from typing import Any, Tuple

import gym
import numpy as np


class BinaryWrapper(gym.Wrapper):
    """
    A wrapper around an environment making it usable with binary networks.

    Only usable as an interface.
    Children must implement `action_dims`, `binarize_observation`, and `get_action`.
    """

    action_dims: int

    def __init__(self, env):
        super(BinaryWrapper, self).__init__(env)

    def reset(self, **kwargs) -> np.ndarray:
        observation = self.env.reset(**kwargs)
        return self.binarize_observation(observation)

    def step(self, binary_action: np.ndarray) -> Tuple[np.ndarray, float, bool, Any]:
        action = self.unbinarize_action(binary_action)
        observation, reward, done, info = self.env.step(action)
        return self.binarize_observation(observation), reward, done, info

    def binarize_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Convert an observation into a valid input to a binary network.

        Transforms `observation` into a binary vector.

        Parameters
        ----------
        observation : np.ndarray
            Observation vector output by the environment.

        Returns
        -------
        np.ndarray
            An array of bools, acting as input to the binary network.
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


class LunarLanderV2(BinaryWrapper):
    action_dims = 4

    # TODO: verify that the log spaces are needed (add both on param?)

    # Bits 0-9 encode X position
    _x_pos_space = np.concatenate(
        [-np.geomspace(0.5, 0.05, 4), [0], np.geomspace(0.05, 0.5, 4)]
    )

    # Bits 10-19 encode Y position
    _y_pos_space = np.linspace(0.15, 1.4, 9)

    # Bits 20-30 encode X velocity
    _x_vel_space = np.concatenate([-np.geomspace(1, 0.01, 5), np.geomspace(0.01, 1, 5)])

    # Bits 31-41 encode Y velocity
    _y_vel_space = np.linspace(-1.5, 1, 10)

    # Bits 42-51 encode angle
    _angle_space = np.linspace(-1.5, 1.5, 9)

    # Bits 52-61 encode angular velocity
    _angle_v_space = np.concatenate(
        [-np.geomspace(1, 0.01, 4), [0], np.geomspace(0.01, 1, 4)]
    )

    # (and bits 62/63 encode whether the left or right foot was touching)

    def __init__(self):
        env = gym.make("LunarLander-v2")
        super(LunarLanderV2, self).__init__(env)

    def binarize_observation(self, observation: np.ndarray) -> np.ndarray:
        x_pos, y_pos, x_vel, y_vel, angle, angle_v, touch_l, touch_r = observation

        x_pos_i = np.searchsorted(self._x_pos_space, x_pos, side="left")
        y_pos_i = np.searchsorted(self._y_pos_space, y_pos, side="left") + 10
        x_vel_i = np.searchsorted(self._x_vel_space, x_vel, side="left") + 20
        y_vel_i = np.searchsorted(self._y_vel_space, y_vel, side="left") + 31
        angle_i = np.searchsorted(self._angle_space, angle, side="left") + 42
        angle_v_i = np.searchsorted(self._angle_v_space, angle_v, side="left") + 52

        discretized = np.zeros(64, bool)
        discretized[x_pos_i] = True
        discretized[y_pos_i] = True
        discretized[x_vel_i] = True
        discretized[y_vel_i] = True
        discretized[angle_i] = True
        discretized[angle_v_i] = True
        discretized[62] = touch_l
        discretized[63] = touch_r

        return discretized

    def unbinarize_action(self, binary_action: np.ndarray) -> int:
        return np.argmax(binary_action)
