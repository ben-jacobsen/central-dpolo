"""
    File: sequences.py
    Author: Ben Jacobsen
    Purpose: Provides classes to encapsulate different strategies for online
        continual counting, represented as sequences of coefficients
"""

from functools import cache

import mpmath as mp
import numpy as np
from numpy.fft import fft, ifft
from scipy.optimize import minimize_scalar

import singular_approximation as sa


class Sequence:

    def __init__(self):
        self._seq = np.array([])
        self.size = 0
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

    def sensitivity(self, k=None):
        """
        Returns the l2 norm of the right sequence, for the purpose of
        deciding how much noise needs to be added to guarantee Gaussian DP

        optionally includes time horizon
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

    def noise_schedule(self, k=None, smooth=False):
        if k is None:
            k = self.size

        if smooth:
            return self.smooth_sensitivity(k) * self.standard_error(k)
        return self.sensitivity(k) * self.standard_error(k)


class Opt(Sequence):

    def __init__(self):
        super().__init__()
        # self.size = inputsize
        self.name = "Sqrt Matrix"

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
            self._seq = np.concatenate(
                (self._seq,
                 np.array([Opt.coeff(n) for n in range(self.size, k)])))
            self.size = k
        return self._seq[:k]

    def first_k_left(self, k):
        return self.first_k(k)

    def sensitivity(self, k=None):
        if k is None:
            k = self.size
        if k <= 1e5:
            return np.linalg.norm(self.first_k(k))
        else:  # avoid exponential blowups with upper bound
            return np.sqrt(1 + np.log(4 * k - 3) / np.pi)


class Independent(Sequence):

    def __init__(self):
        super().__init__()
        self.name = "Ind"

    def first_k(self, k):
        if k > self.size:
            self._seq = np.zeros(k)
            self._seq[0] = 1
            self.size = k
        return self._seq[:k]

    def first_k_left(self, k):
        return np.ones(k)

    def sensitivity(self, k=None):
        return 1


class Anytime(Sequence):

    def __init__(self, alpha, gamma, delta=None, tol=0, asym_order=5):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.tol = tol  # threshold to switch to asymptotic expansion
        self.approximating = False
        self.asym_order = asym_order
        if delta is None:
            self.delta = -6 * gamma / 5
        else:
            self.delta = delta

        self.name = f"γ={gamma:.2f}"
        if delta != 0:
            self.name += f", δ={delta:.2f}"
        if self.tol != 0:
            self.name += f", η={tol:.0e}"

        self._left_seq = np.array([])

    def first_k(self, k):
        if k > self.size:
            newsize = max(k, self.size * 2)
            self._grow(newsize)

        return self._seq[:k]

    def first_k_left(self, k):
        if k > self.size:
            newsize = max(k, self.size * 2)
            self._grow(newsize)

        if self._left_seq.size < self.size:
            self._left_seq = np.cumsum(fast_inv_ltt(self._seq))

        return self._left_seq[:k]

    def _grow(self, newsize):
        might_approx = self.tol > 0 and not self.approximating
        while might_approx and newsize > (int_step := min(
                max(10000, self.size * 2), newsize)):
            print(f"{newsize} >>> {self.size}, trying {int_step} first...")
            self._grow(int_step)
            might_approx = not self.approximating

        # check if we can switch to asymptotics
        if self.tol > 0 and self.size > 0 and not self.approximating:
            estimate = sa.full_computation(self.size - 1,
                                           gamma=self.gamma,
                                           K=self.asym_order,
                                           delta=self.delta,
                                           single_coeff=True)
            err = abs(
                max(estimate[0] /
                    self._seq[self.size - 1], self._seq[self.size - 1] /
                    estimate[0]) - 1)
            print(f"error: {err}")
            if err < self.tol:
                print("Switching to asymptotics")
                self.approximating = True

        if self.approximating:
            asymptotic_seq = sa.full_computation(newsize,
                                                 gamma=self.gamma,
                                                 K=self.asym_order,
                                                 delta=self.delta)
            self._seq = np.concatenate((self._seq, asymptotic_seq[self.size:]))
        else:
            self._seq = sa.exact_convolution(newsize, self.gamma, self.alpha,
                                             self.delta)

        # self._left_seq = np.cumsum(fast_inv_ltt(self._seq))
        self.size = newsize

    def sensitivity(self, k=None):
        return (1 + self.tol) * float(sa.compute_sensitivity(self.alpha, self.gamma,
                                            self.delta))

    def optimize_delta(self, T, **kwargs):
        """
        Given a *fixed* time horizon and gamma/alpha, find the delta parameter
        minimizing actual variance at the final time step
        """

        def sens(delta):
            return float(sa.compute_sensitivity(self.alpha, self.gamma, delta))

        # def sens_growth_est(delta):
        #     n = sym.Symbol('n')
        #     d = sym.Symbol('d')
        #     f = 4**d / np.pi * (1 / n) * (sym.log(n))**(-2 * self.gamma) * (
        #         sym.log(sym.log(n)))**(-2 * d)

        def se(delta):
            return Anytime(self.alpha, -self.gamma,
                           -delta).smooth_sensitivity(T)[-1]

        return minimize_scalar(lambda delta: sens(delta) * se(delta) /
                               (-self.gamma),
                               **kwargs,
                               options={'disp': True})


class BinaryMechanism:

    def __init__(self, totalsize=0):
        self.size = totalsize
        self.name = "Binary"

    @staticmethod
    def _count_bits(t):
        total = 0
        for b in bin(t):
            total += b == '1'
        return total

    def noise_schedule(self, k):
        if k > self.size:
            self.size = k
        return np.sqrt(np.floor(np.log2(self.size)) + 1) * np.sqrt(np.array([BinaryMechanism._count_bits(t) for t in range(1,k+1)]))


class SmoothBinary:

    def __init__(self, totalsize=0):
        self.size = totalsize
        self.name = "Smooth Binary"


    def noise_schedule(self, k):
        if k > self.size:
            self.size = k
        return np.ones(k) / 2 * (np.log(self.size) + np.log(np.log(self.size)))




class DoublingTrick:

    def __init__(self, init_chunk, totalsize, exponential=False, ratio=2):
        self._subseqs = []
        self.size = 0
        self._next_chunk = init_chunk
        self.ratio = ratio
        self.is_exp = exponential
        self.grow(totalsize)

        self.name = f"Doubling Trick"

    def grow(self, newsize):
        while newsize - self.size > 0:
            self._subseqs.append(self._next_chunk)
            self.size += self._next_chunk
            if self.is_exp:
                self._next_chunk = self._next_chunk**2
            else:
                self._next_chunk = int(self._next_chunk * self.ratio)

    def noise_schedule(self, k=None):
        if k is None:
            k = self.size
        elif k > self.size:
            self.grow(k)
        schedule = np.zeros(k)
        i = 0
        extra_noise = 0

        o = Opt()
        for subseq in self._subseqs:
            remaining = min(k - i, subseq)
            if remaining <= 0:
                break

            i_next = i + remaining
            schedule[i:i_next] = (o.standard_error(remaining) * o.sensitivity(subseq))**2 + extra_noise

            extra_noise += schedule[i_next - 1]
            i = i_next

        return np.sqrt(schedule)

    @staticmethod
    def optimal_ratio():
        return float(mp.findroot(lambda b: b**1.5 - 2 * b + 1, 2))

class Hybrid:
    """
    Hybrid algorithm, employing an Anytime algorithm that sums the condensed sequence

    y_0 = x_0
    \sum_{i=0}^k y_k = \sum_{j=0}^{2^k-1} x_j

    or, expressed differently,

    y_k = \sum_{j=2^{k-1}}^{2^k - 1} x_j

    alongside a sequence of Optimal algorithms that sum all of the values
    between the y_i releases. 
    """

    def __init__(self, at=None, bounded=None, init_chunk=2, w=0.5, ratio=2, exponential=False):
        """
        alpha, gamma, delta are parameters of the Anytime algorithm, while
        w controls the portion of the privacy budget allocated to the
        Anytime algorithm
        """
        if at is None:
            at = Anytime(alpha=-1/2, gamma=-0.55, delta=0)
        if bounded is None:
            bounded = Opt()
        self._at = at
        self._bounded = bounded
        self._subseqs = []
        self.size = 0
        self.w = w
        self._next_chunk = init_chunk
        self.ratio = ratio
        self.exponential = exponential
        self.name = f"Hybrid ({at.name})"


    def grow(self, newsize):
        while (remaining := newsize - self.size) > 0:
            self._subseqs.append(self._next_chunk)
            self.size += self._next_chunk 

            if self.exponential:
                self._next_chunk = self._next_chunk**2
            else:
                self._next_chunk = self.ratio * self._next_chunk



    def noise_schedule(self, k=None):
        if k is None:
            k = self.size
        if k > self.size:
            self.grow(k)

        schedule = np.zeros(k)
        start_index = 0
        acc_local_var = 0
        for i, subseq in enumerate(self._subseqs):
            stop_index = min(start_index + subseq, k)
            # the chunk we're looking at
            # anytime error
            at_var = self._at.noise_schedule(i+1)[i]**2 / self.w
            # reuse the Doubling Trick info
            combined_var = 2 * acc_local_var * at_var / (np.sqrt(at_var) + np.sqrt(acc_local_var))**2
            schedule[start_index:stop_index] = combined_var

            # subsequence error
            diff = stop_index - start_index
            local_var = (self._bounded.noise_schedule(subseq))**2 / (1-self.w)
            print(local_var)
            schedule[start_index:stop_index] += local_var[:diff]

            start_index = stop_index
            acc_local_var += local_var[-1]

        return np.sqrt(schedule)


    @staticmethod
    def optimize_weight(T):
        """ 
        Cheat a little bit by calibrating weight to time horizon
        """
        return 1/(1 + (1 + np.log(T))/(1 + np.log(np.log2(T))))






def fast_inv_ltt(a):
    n = len(a)
    eps = 10**(-5 / n)
    vec = np.power(eps, np.arange(n))
    return (ifft(1 / fft(a * vec)) / vec).real
