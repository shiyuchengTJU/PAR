"""
Noise generation from Normal distributions.
"""

import numpy as np


def sample_hypersphere(n_samples, sample_shape, radius, l_norm=2, mode='sphere', sample_gen=None, seed=None):
    """
    Uniformly sample the surface of a hypersphere.
    Uniform picking: create a n-dimensional normal distribution and then normalize it to the desired radius.
    See http://mathworld.wolfram.com/HyperspherePointPicking.html
    WARNING: this is probably not correct for other norms!! We should check it out carefully if we don't use L2.
    :param n_samples: number of image samples to generate.
    :param sample_shape: shape of a single image sample.
    :param radius: radius(=eps) of the hypersphere.
    :param l_norm: L-norm.
    :param mode: if 'sphere', then samples the surface of the eps-sphere. If 'ball', then samples the volume of the eps-ball.
                Note: 'ball' is currently unused, and certainly not uniformly distributed.
    :param sample_gen: If provided, retrieves random numbers from this generator.
    :param seed: seed for the random generator. Cannot be used with the sample generator.
    :return: Batch of image samples, shape: (n_samples,) + sample_shape
    """

    if sample_gen is not None:
        assert seed is None, "Can't provide individual seeds if using the multi-threaded generator."
        assert sample_shape == sample_gen.shape

        # Get precalculated samples from the generator
        gauss = np.empty(shape=(n_samples, np.prod(sample_shape)), dtype=np.float64)
        for i in range(n_samples):
            gauss[i] = sample_gen.get_normal().reshape(-1)
    else:
        if seed is not None:
            np.random.seed(seed)
        gauss = np.random.normal(size=(n_samples, np.prod(sample_shape)))

    # Norm to
    norm = np.linalg.norm(gauss, ord=l_norm, axis=1)
    perturbation = (gauss / norm[:, np.newaxis])

    # Sphere: sample only the surface of the hypersphere.
    # Ball: sample inside the sphere. Note: this is probably not uniform.
    if mode == 'sphere':
        perturbation *= radius
    elif mode == 'ball':
        perturbation *= np.random.uniform(low=0.0, high=radius, size=(n_samples, 1))
    else:
        raise ValueError("Unknown sampling mode.")

    perturbation = np.reshape(perturbation, (n_samples,) + sample_shape)

    return perturbation