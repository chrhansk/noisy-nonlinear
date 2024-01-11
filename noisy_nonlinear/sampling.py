import numpy as np


def sample_sphere(random_state,
                  dim,
                  radius):
    """
    Sample a point on the sphere of radius `radius` in `dim` dimensions.
    """
    assert dim > 0
    assert radius >= 0.

    sample = random_state.normal(size=(dim,))

    norm = np.linalg.norm(sample, ord=2)
    assert norm != 0.

    return (radius / norm) * sample


def sample_ball(random_state,
                dim,
                radius):
    """
    Sample a point in the ball of radius `radius` in `dim` dimensions.
    """
    assert dim > 0
    assert radius >= 0.

    sample = random_state.normal(size=(dim,))

    norm = np.linalg.norm(sample, ord=2)
    assert norm != 0.

    factor = random_state.uniform(0., 1.) ** (1. / dim)
    factor *= (radius / norm)

    return factor * sample


def sample_uniform(random_state,
                   low,
                   high,
                   size):
    """
    Sample a point in the range [low, high] in `dim` dimensions.
    """

    assert low <= high

    return random_state.uniform(low=low,
                                high=high,
                                size=size)
