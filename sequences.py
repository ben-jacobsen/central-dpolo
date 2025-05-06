"""
    File: sequences.py
    Author: Ben Jacobsen
    Purpose: Provides classes to encapsulate different strategies for online
        continual counting, represented as sequences of coefficients
"""

from functools import cache

import mpmath as mp
import numpy as np
from numpy.fft import *

import singular_approximation as sa


class Sequence:

    def __init__(self):
        self._seq = np.array([1])
        self.size = 1
        self.name = "Mystery Sequence"

    def first_k(self, k):
        """
        Returns a numpy array with the first k elements of the sequence
        """
        raise NotImplementedError

    def first_k_left(self, k):
        """
        Return a numpy array with the first k elements of the 'left' sequence,
        meaning the sequence that solve the equation:

            conv(L, R) = 1,1,1,1,...
        """
        raise NotImplementedError

    def sensitivity(self):
        """
        Returns the l2 norm of the right sequence, for the purpose of
        deciding how much noise needs to be added to guarantee Gaussian DP
        """
        raise NotImplementedError

    def standard_error(self, k=None):
        """
        Returns the accumulated l2 norm of the left sequence, for the purpose
        of estimating error as a function of time
        """
        if k is None:
            k = self.size
        return np.sqrt(np.cumsum(np.pow(self.first_k_left(k),
                                        2))).astype(float)

    def smooth_sensitivity(self, k=None):
        """
        Returns the accumulated l2 norm of the right sequence. Useful only for
        analysis, not as a true sensitivity
        """
        if k is None:
            k = self.size
        return np.sqrt(np.cumsum(np.pow(self.first_k(k), 2))).astype(float)

    def noise_schedule(self, k=None):
        if k is None:
            k = self.size

        return self.sensitivity() * self.standard_error(k)


class Opt(Sequence):

    def __init__(self, inputsize):
        self.size = inputsize
        self._seq = np.array([Opt.coeff(n) for n in range(inputsize)])
        self.name = "Optimal"

    @staticmethod
    @cache
    def coeff(k):
        """
        Computes the kth coefficient of the sequence recursively
        """
        if k == 0:
            return 1

        return (1 - 1 / (2 * k)) * Opt.coeff(k - 1)

    def first_k(self, k):
        if k > self.size:
            raise ValueError(
                f"Can't return first {k} from sequence of size {self.size}!")
        return self._seq[:k]

    def first_k_left(self, k):
        return self.first_k(k)

    def sensitivity(self):
        return np.linalg.norm(self._seq)


class Anytime(Sequence):

    def __init__(self, alpha, gamma, delta):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta

        self.name = f"Logarithmic (gamma={gamma:.2f}"
        if delta != 0:
            self.name += f", delta={delta:.2f}"
        self.name += ")"

        self._left_seq = np.array([1])

    def first_k(self, k):
        if k > self.size:
            newsize = max(k, self.size * 2)
            self._grow(newsize)

        return self._seq[:k]

    def first_k_left(self, k):
        if k > self.size:
            newsize = max(k, self.size * 2)
            self._grow(newsize)

        return self._left_seq[:k]

    def _grow(self, newsize):
        self._seq = sa.exact_convolution(newsize, self.gamma, self.alpha,
                                         self.delta)
        self._left_seq = np.cumsum(fast_inv_ltt(self._seq))
        self.size = newsize

    def sensitivity(self):
        return float(sa.compute_sensitivity(self.alpha, self.gamma,
                                            self.delta))

    def optimize_delta(self):
        super().__init__()
        self.delta = sa.optimize_sensitivity(self.alpha, self.gamma)


class DoublingTrick:

    def __init__(self, init_chunk, ratio, totalsize):
        self._subseqs = []
        self.size = 0
        self._next_chunk = init_chunk
        self.ratio = ratio
        self.grow(totalsize)

        self.name = f"Doubling Trick (chunk size {init_chunk}, ratio {ratio:.2f})"

    def grow(self, newsize):
        while newsize - self.size > 0:
            self._subseqs.append(Opt(self._next_chunk))
            self.size += self._next_chunk
            self._next_chunk = int(self._next_chunk * self.ratio)

    def noise_schedule(self, k=None):
        if k is None:
            k = self.size
        elif k > self.size:
            self.grow(k)
        schedule = np.zeros(k)
        i = 0
        extra_noise = 0

        for subseq in self._subseqs:
            remaining = min(k - i, subseq.size)
            if remaining <= 0:
                break

            i_next = i + remaining
            schedule[i:i_next] = subseq.noise_schedule(remaining) + extra_noise

            extra_noise += schedule[i_next - 1]
            i = i_next

        return schedule

    @staticmethod
    def optimal_ratio():
        return float(mp.findroot(lambda b: b**1.5 - 2 * b + 1, 2))


def fast_inv_ltt(a):
    n = len(a)
    eps = 10**(-5 / n)
    vec = np.power(eps, np.arange(n))
    return (ifft(1 / fft(a * vec)) / vec).real
