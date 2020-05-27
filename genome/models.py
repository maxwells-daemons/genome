from typing import List, Optional, Tuple, Union

import binary_layers
import numpy as np
import scipy.special
from genome import binary_layers_numpy

BinaryLayer = Union[binary_layers.BinaryLayer64, binary_layers.LinearLayer64]


class LayerDistribution64:
    """
    The template for a binary dense layer with 64 units.

    Represents a probability distribution over the weights and biases of a dense
    layer in a binary neural network. All parameters are independently distributed;
    weights have a Bernoulli distribution and biases have a Gaussian distribution.
    Can be sampled or binarized to get a binary layer, and trained to shift the
    distribution towards high-performing samples.

    Primarily useful as a component of NetworkDistribution64. Discretizes to either a
    BinaryLayer64 or a LinearLayer64, depending on whether `output` is set.

    Parameters
    ----------
    output : bool
        True if this is an output layer, otherwise False.
        If it is, then it is linear.
    num_outputs : Optional[int] (default: None)
        If `output` is True, the number of outputs for the layer, and must be provided.
        Otherwise, unused.
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

    LAYER_WIDTH = 64
    EPSILON = 1e-4

    output: bool
    use_bias: bool
    num_outputs: int
    weight_logits: np.ndarray
    bias_mean: np.ndarray
    bias_std: np.ndarray
    fix_bias_std: bool

    def __init__(
        self,
        output: bool,
        num_outputs: Optional[int] = None,
        init_weight_logits_std: float = 0,
        use_bias: bool = False,
        fixed_bias_std: Optional[float] = None,
        init_bias_std: float = 1.0,
    ):
        self.output = output
        self.use_bias = use_bias

        if output:
            assert num_outputs is not None
            self.num_outputs = num_outputs
            self.binary_type = binary_layers.LinearLayer64
        else:
            self.num_outputs = self.LAYER_WIDTH
            self.binary_type = binary_layers.BinaryLayer64

        self.weight_logits = np.random.normal(
            0.0, init_weight_logits_std, (self.LAYER_WIDTH, self.num_outputs)
        )

        if use_bias:
            self.bias_mean = np.zeros(self.num_outputs)
            self.bias_std = np.full(self.num_outputs, fixed_bias_std or init_bias_std)
            self.fix_bias_std = fixed_bias_std is not None

    def sample(self, _as_numpy: bool = False) -> BinaryLayer:
        """
        Sample a binary layer from this distribution.

        Sample binary weights from the Bernoulli distribution defined by `weight_logits`
        and integer biases from the Gaussian distribution defined by `bias_mean` and
        `bias_std`.

        Parameters
        ----------
        _as_numpy : bool (default: False)
            If True, return a layer implemented in numpy instead of efficient Cython.
            Used for debugging only.

        Returns
        -------
        BinaryLayer
            A Cython extension class implementing the sampled layer.
            Generally only useful as part of a BinaryNetwork64.
        """
        probs = scipy.special.expit(self.weight_logits)
        weights = np.random.binomial(1, probs).astype(bool)

        if self.use_bias:
            biases = (
                np.random.normal(self.bias_mean, self.bias_std).round().astype(np.int32)
            )
        else:
            biases = np.zeros(self.num_outputs, np.int32)

        if _as_numpy:
            if self.binary_type == binary_layers.LinearLayer64:
                return binary_layers_numpy.LinearLayer64(weights, biases)
            return binary_layers_numpy.BinaryLayer64(weights, biases)
        return self.binary_type.__new__(self.binary_type, weights, biases)

    def binarize_maximum_likelihood(self, _as_numpy: bool = False) -> BinaryLayer:
        """
        Compute the maximum-probability binary layer under this distribution.

        Deterministically pick weights and biases that maximimize the likelihood under
        this distribution.

        Parameters
        ----------
        _as_numpy : bool (default: False)
            If True, return a layer implemented in numpy instead of efficient Cython.
            Used for debugging only.

        Returns
        -------
        BinaryLayer
            A Cython extension class implementing the maximum-likelihood layer.
            Generally only useful as part of a BinaryNetwork64.
        """
        weights = self.weight_logits > 0
        if self.use_bias:
            biases = self.bias_mean.round().astype(np.int32)
        else:
            biases = np.zeros(self.num_outputs, np.int32)

        if _as_numpy:
            if self.binary_type == binary_layers.LinearLayer64:
                return binary_layers_numpy.LinearLayer64(weights, biases)
            return binary_layers_numpy.BinaryLayer64(weights, biases)
        return self.binary_type.__new__(self.binary_type, weights, biases)

    def step(
        self,
        learning_rate: float,
        samples_and_returns: List[Tuple[BinaryLayer, float]],
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
        samples_and_returns : [(BinaryLayer, float)]
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

        for sample, value in samples_and_returns:
            weights, biases = sample.get_params()
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

        if self.use_bias:
            self.bias_mean += learning_rate * bias_mean_gradient / bias_variance

            if not self.fix_bias_std:
                self.bias_std += learning_rate * bias_std_gradient / bias_std_cubed
                self.bias_std = np.maximum(
                    self.bias_std, np.full_like(self.bias_std, self.EPSILON)
                )


class NetworkDistribution64:
    """
    The template for a binary neural network with 64-dimensional activations.

    Represents a collection of probability distributions, one for each layer.
    Together this forms a joint distribution over binary networks.
    Discretizes to a BinaryNetwork64.

    Parameters
    ----------
    num_hidden_layers : int
        The number of hidden layers for the network.
    num_outputs : int
        The number of integer-valued outputs for the network.
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
    hidden_layers: List[LayerDistribution64]
    output_layer: LayerDistribution64

    def __init__(
        self,
        num_hidden_layers: int,
        num_outputs: int,
        init_weight_logits_std: float = 0,
        use_bias: bool = False,
        fixed_bias_std: Optional[float] = None,
        init_bias_std: float = 1,
    ):
        self.hidden_layers = [
            LayerDistribution64(
                output=False,
                num_outputs=None,
                init_weight_logits_std=init_weight_logits_std,
                use_bias=use_bias,
                fixed_bias_std=fixed_bias_std,
                init_bias_std=init_bias_std,
            )
            for _ in range(num_hidden_layers)
        ]

        self.output_layer = LayerDistribution64(
            output=True,
            num_outputs=num_outputs,
            init_weight_logits_std=init_weight_logits_std,
            use_bias=use_bias,
            fixed_bias_std=fixed_bias_std,
            init_bias_std=init_bias_std,
        )

    def sample(self, _as_numpy: bool = False) -> binary_layers.BinaryNetwork64:
        """
        Sample a binary network from this distribution.

        Each layer is sampled independently according to its distribution.

        Parameters
        ----------
        _as_numpy : bool (default: False)
            If True, return a layer implemented in numpy instead of efficient Cython.
            Used for debugging only.

        Returns
        -------
        BinaryNetwork64
            A Cython extension class implementing the sampled network.
        """
        hidden = [layer.sample(_as_numpy) for layer in self.hidden_layers]
        output = self.output_layer.sample(_as_numpy)
        if _as_numpy:
            return binary_layers_numpy.BinaryNetwork64(hidden, output)
        return binary_layers.BinaryNetwork64.__new__(
            binary_layers.BinaryNetwork64,
            np.array(hidden, dtype=binary_layers.BinaryLayer64),
            output,
        )

    def binarize_maximum_likelihood(self, _as_numpy: bool = False):
        """
        Compute the maximum-probability binary network under this distribution.

        Works by picking each layer according to the maximum likelihood under its
        distribution.

        Parameters
        ----------
        _as_numpy : bool (default: False)
            If True, return a layer implemented in numpy instead of efficient Cython.
            Used for debugging only.

        Returns
        -------
        BinaryNetwork64
            A Cython extension class implementing the sampled network.
        """
        hidden = [
            layer.binarize_maximum_likelihood(_as_numpy) for layer in self.hidden_layers
        ]
        output = self.output_layer.binarize_maximum_likelihood(_as_numpy)
        if _as_numpy:
            return binary_layers_numpy.BinaryNetwork64(hidden, output)
        return binary_layers.BinaryNetwork64.__new__(
            binary_layers.BinaryNetwork64,
            np.array(hidden, dtype=binary_layers.BinaryLayer64),
            output,
        )

    def step(
        self,
        learning_rate: float,
        samples_and_returns: List[Tuple[binary_layers.BinaryNetwork64, float]],
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
        samples_and_returns : [(BinaryNetwork64, float)]
            A list of each test point and the value it obtained.
        natural_gradient : bool (default: True)
            If True, use the natural gradient update for the weight logits.
            Otherwise, use the plain gradient.
        """
        # Shape: [num_hidden_layers, num_samples]
        hidden_layers_and_returns: List[List[binary_layers.BinaryLayer64]] = [[]] * len(
            self.hidden_layers
        )
        output_layers_and_returns = []

        for network, ret in samples_and_returns:
            for i, layer in enumerate(network.hidden_layers):
                hidden_layers_and_returns[i].append((layer, ret))
            output_layers_and_returns.append((network.output_layer, ret))

        for distribution, sample in zip(self.hidden_layers, hidden_layers_and_returns):
            distribution.step(learning_rate, sample, natural_gradient)
        self.output_layer.step(
            learning_rate, output_layers_and_returns, natural_gradient
        )
