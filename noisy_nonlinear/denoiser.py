import numpy as np
from PIL import Image
from math import sqrt

from noisy_nonlinear.func import Func
from noisy_nonlinear.sampling import sample_uniform

default_seed = 42


class Denoiser(Func):
    def __init__(self, filename, noise_level=.1, **kwds):
        self.image = np.asarray(Image.open(filename))
        self.image = self.image.astype(float) / 255.

        seed = kwds.get('seed', default_seed)

        self.filename = filename
        self.noise_level = noise_level
        self._rng = np.random.RandomState(seed=seed)

    def _noisy_image(self):
        level = self.noise_level

        noise = sample_uniform(self._rng,
                               low=-level,
                               high=level,
                               size=self.image.shape)

        return np.clip(self.image + noise, 0., 1.)

    def as_image(self, x):
        x_int = (x * 255).astype(np.uint8)
        return Image.fromarray(x_int.reshape(self.image.shape))

    def value(self, x):
        noisy_image = self._noisy_image()

        guess = x.reshape(self.image.shape)

        delta = guess - noisy_image
        norm = np.linalg.norm(delta)
        fidelity = .5 * norm*norm

        return np.array([fidelity])

    def opt(self):
        return self.image.ravel()

    def deriv_values(self, x):
        guess = x.reshape(self.image.shape)

        noisy_image = self._noisy_image()

        return (guess - noisy_image).ravel()

    def deriv_struct(self):
        return Func.full_indices((1, self.dim()))

    def value_error(self):
        noise_factor = (self.noise_level + .5 * (self.noise_level)**2)
        return noise_factor * self.image.size

    def deriv_error(self):
        return self.noise_level * sqrt(self.image.size)

    def deriv_lipschitz(self, pmin, pmax):
        return 1.

    def num_cons(self):
        return 0

    def dim(self):
        return self.image.size

    def orig_func(self):
        return self
