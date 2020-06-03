from typing import List, Optional, Tuple

import itertools as it
import numpy as np
import scipy.special

from genome.binary_networks import inference

# From: https://docs.python.org/3/library/itertools.html
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)


class LayerDistribution:
    """
    The template for a binary dense layer.

    Represents a probability distribution over the weights and biases of a dense
    layer in a binary neural network. All parameters are independently distributed;
    weights have a Bernoulli distribution and biases have a Gaussian distribution.
    Can be sampled or binarized to get a binary layer, and trained to shift the
    distribution towards high-performing samples.

    Primarily useful as a component of NetworkDistribution. Discretizes to LayerParams.

    Parameters
    ----------
    input_dims : int
        The number of inputs for this layer.
    output_dims : int
        The number of outputs for this layer.
    init_weight_logits_std : float (default: 0)
        The standard deviation used to initialize the weight logits.
        When zero (default), all of the weight logits are initialized to 0.
    use_bias : bool (default: False)
        If False, always use a bias of zero.
    fixed_bias_std : Optional[float] (default: None)
        If provided, the bias terms will always be sampled with this standard deviation,
        which will not be adapted.
    init_bias_std : float (default: 1)
        The initial standard deviation of the bias distribution.
        If fixed_bias_std is provided, this parameter is unused.
    """

    EPSILON = 0.1

    input_dims: int
    output_dims: int
    use_bias: bool
    fix_bias_std: bool

    weight_logits: np.ndarray
    bias_mean: np.ndarray
    bias_std: np.ndarray

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        init_weight_logits_std: float = 0,
        use_bias: bool = False,
        fixed_bias_std: Optional[float] = None,
        init_bias_std: float = 1.0,
    ):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.use_bias = use_bias

        self.weight_logits = np.random.normal(
            0.0, init_weight_logits_std, (self.input_dims, self.output_dims)
        )

        if use_bias:
            self.bias_mean = np.zeros(self.output_dims)
            self.bias_std = np.full(self.output_dims, fixed_bias_std or init_bias_std)
            self.fix_bias_std = fixed_bias_std is not None

    def sample(self) -> inference.LayerParams:
        """
        Sample a binary layer from this distribution.

        Sample binary weights from the Bernoulli distribution defined by `weight_logits`
        and integer biases from the Gaussian distribution defined by `bias_mean` and
        `bias_std`.

        Returns
        -------
        LayerParams
            Discrete parameters of the sampled layer.
        """
        probs = scipy.special.expit(self.weight_logits)
        weights = np.random.binomial(1, probs).astype(bool)

        if self.use_bias:
            biases = (
                np.random.normal(self.bias_mean, self.bias_std).round().astype(np.int32)
            )
        else:
            biases = np.zeros(self.output_dims, np.int32)

        return (weights, biases)

    def binarize_maximum_likelihood(self) -> inference.LayerParams:
        """
        Compute the maximum-probability binary layer under this distribution.

        Deterministically pick weights and biases that maximimize the likelihood under
        this distribution.

        Returns
        -------
        LayerParams
            Discrete parameters of the MAP layer.
        """
        weights = self.weight_logits > 0
        if self.use_bias:
            biases = self.bias_mean.round().astype(np.int32)
        else:
            biases = np.zeros(self.output_dims, np.int32)

        return (weights, biases)

    def step(
        self,
        learning_rate: float,
        samples_and_returns: List[Tuple[inference.LayerParams, float]],
        natural_gradient: bool = True,
    ) -> None:
        """
        Update this layer's distribution towards high performers in a batch of test
        points.

        Takes a step of simple SGD to maximize the "parameter-exploring policy
        gradient" (log probability of each parameter multiplied by that samples return)
        with evolution strategies. Uses the parameter gradient, not the natural
        gradient.

        Parameters
        ----------
        learning_rate : float
            The learning rate to use for this step.
        samples_and_returns : [(LayerParams, float)]
            A list of each test point and the value it obtained.
        natural_gradient : bool (default: True)
            If True, use the natural gradient update for the weight logits.
            Otherwise, use the plain gradient.
        """

        learning_rate /= len(samples_and_returns)
        current_probs = scipy.special.expit(self.weight_logits)
        weight_gradient = np.zeros_like(self.weight_logits)

        if self.use_bias:
            bias_mean_gradient = np.zeros_like(self.bias_mean)
            bias_variance = np.power(self.bias_std, 2.0)

            if not self.fix_bias_std:
                bias_std_gradient = np.zeros_like(self.bias_std)
                bias_std_cubed = np.power(self.bias_std, 3.0)

        for (weights, biases), value in samples_and_returns:
            weights = weights.astype(float)
            biases = biases.astype(float)

            weight_gradient += value * (weights - current_probs)

            if self.use_bias:
                bias_deviation = biases - self.bias_mean
                bias_mean_gradient += value * bias_deviation

                if not self.fix_bias_std:
                    bias_std_gradient += value * (
                        np.power(bias_deviation, 2) - bias_variance
                    )

        if natural_gradient:
            fisher_information = current_probs * (1 - current_probs)
            weight_gradient /= fisher_information + self.EPSILON

        # TODO: try optimizers other than plain SGD
        self.weight_logits += learning_rate * weight_gradient

        # TODO: comment, set as hparam
        # self.weight_logits *= 0.99

        if self.use_bias:
            self.bias_mean += learning_rate * bias_mean_gradient / bias_variance

            if not self.fix_bias_std:
                self.bias_std += learning_rate * bias_std_gradient / bias_std_cubed
                self.bias_std = np.maximum(
                    self.bias_std, np.full_like(self.bias_std, self.EPSILON)
                )


class NetworkDistribution:
    """
    The template for a binary neural network.

    Represents a collection of probability distributions, one for each layer.
    Together this forms a joint distribution over binary networks.
    Discretizes to NetworkParams.

    Parameters
    ----------
    layer_dims : [int]
        A list of all of the widths that will appear in the network. The first element
        is the input dimensionality, and the last is the number of outputs.
    init_weight_logits_std : float (default: 0)
        The standard deviation used to initialize the weight logits of each
        layer. When zero (default), all of the weight logits are initialized
        to 0.5.
    use_bias : bool (default: False)
        If False, layers always use a bias of zero.
    fixed_bias_std : Optional[float] (default: None)
        If provided, the bias terms will always be sampled with this standard deviation,
        which will not be adapted.
    init_bias_std : float (default: 1)
        The initial standard deviation of the bias distribution.
        If fixed_bias_std is provided, this parameter is unused.
    """

    use_bias: bool
    layers: List[LayerDistribution]
    layer_dims: List[int]

    def __init__(
        self,
        layer_dims: List[int],
        init_weight_logits_std: float = 0,
        use_bias: bool = False,
        fixed_bias_std: Optional[float] = None,
        init_bias_std: float = 1,
    ):
        self.layer_dims = layer_dims
        self.layers = [
            LayerDistribution(
                input_dims,
                output_dims,
                init_weight_logits_std,
                use_bias,
                fixed_bias_std,
                init_bias_std,
            )
            for input_dims, output_dims in pairwise(layer_dims)
        ]

    def sample(self) -> inference.NetworkParams:
        """
        Sample a binary network from this distribution.

        Each layer is sampled independently according to its distribution.

        Returns
        -------
        NetworkParams
            The parameters of this discrete network sample.
        """
        return [layer.sample() for layer in self.layers]

    def binarize_maximum_likelihood(self):
        """
        Compute the maximum-probability binary network under this distribution.

        Works by picking each layer according to the maximum likelihood under its
        distribution.

        Returns
        -------
        NetworkParams
            The parameters of the MAP approximation to this network.
        """
        return [layer.binarize_maximum_likelihood() for layer in self.layers]

    def step(
        self,
        learning_rate: float,
        samples_and_returns: List[Tuple[inference.NetworkParams, float]],
        natural_gradient: bool = True,
    ) -> None:
        """
        Update this network's distribution towards high performers in a batch of test
        points.

        Each layer takes its own step independently.

        Parameters
        ----------
        learning_rate : float
            The learning rate to use for this step.
        samples_and_returns : [(NetworkParams, float)]
            A list of each test point and the value it obtained.
        natural_gradient : bool (default: True)
            If True, use the natural gradient update for the weight logits.
            Otherwise, use the plain gradient.
        """
        # For every layer, the full batch of discrete samples and returns
        # Shape: [num_hidden_layers, num_samples][2]
        layers_and_returns: List[List[Tuple[inference.LayerParams, float]]] = [
            [] for _ in range(len(self.layers))
        ]
        for network, ret in samples_and_returns:
            for i, layer in enumerate(network):
                layers_and_returns[i].append((layer, ret))

        for distribution, sample_batch in zip(self.layers, layers_and_returns):
            distribution.step(learning_rate, sample_batch, natural_gradient)
