"""
    File: sequences.py
    Author: Ben Jacobsen
    Purpose: Provides classes to encapsulate different strategies for online
        continual counting, represented as sequences of coefficients
"""

from functools import cache

import numpy as np
from numpy.fft import *

import singular_approximation as sa


class Sequence:

    def __init__(self):
        self._seq = np.array([1])
        self._size = 1


    def first_k_right(self, k):
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
            k = self._size
        return np.sqrt(np.cumsum(np.pow(self.first_k_left(k), 2)))



        
class Opt(Sequence):

    def __init__(self, input_size):
        self._size = input_size
        self._seq = np.array([Opt.coeff(n) for n in range(input_size)])


    @staticmethod
    @cache
    def coeff(k):
        """
        Computes the kth coefficient of the sequence recursively
        """
        if k == 0:
            return 1

        return (1 - 1/(2*k)) * Opt.coeff(k-1)

    def first_k_right(self, k):
        if k > self._size:
            raise ValueError(f"Can't return first {k} from sequence of size {self._size}!")
        return self._seq[:k]


    def first_k_left(self, k):
        return self.first_k_right(k)

    def sensitivity(self):
        return np.linalg.norm(self._seq)


class Anytime(Sequence):

    def __init__(self, alpha, gamma, delta):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta

        self._left_seq = np.array([1])


    def first_k_right(self, k):
        if k > self._size:
            new_size = max(k, self._size * 2)
            self._grow(new_size)

        return self._seq[:k]


    def first_k_left(self, k):
        if k > self._size:
            new_size = max(k, self._size * 2)
            self._grow(new_size)

        return self._left_seq[:k]


    def _grow(self, new_size):
        self._seq = sa.exact_convolution(new_size, self.gamma, self.alpha, self.delta)
        self._left_seq = np.cumsum(fast_inv_ltt(self._seq))
        self._size = new_size


    def sensitivity(self):
        return sa.compute_sensitivity(self.alpha, self.gamma, self.delta)

    def optimize_delta(self):
        super().__init__()
        self.delta = sa.optimize_sensitivity(self.alpha, self.gamma)



def fast_inv_ltt(a):
    n = len(a)
    eps = 10**(-5 / n)
    vec = np.power(eps, np.arange(n))
    return (ifft(1 / fft(a * vec)) / vec).real







