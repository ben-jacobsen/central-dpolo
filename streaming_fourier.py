"""
    File: streaming_fourier.py
    Author: Ben Jacobsen
    Purpose: Implements a recursive algorithm for computing the convolution of
        two streams of unknown size in n log n time.
"""

import argparse

import numpy as np
from scipy.signal import convolve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', type=int, default=5)
    args = parser.parse_args()

    size = 2**args.T
    gen = np.random.default_rng()

    s1 = gen.integers(-10, 10, size)
    s2 = gen.integers(-10, 10, size)
    out = np.zeros(size)

    online_out = online_convolve(s1, s2, out)
    exact = convolve(s1, s2)[:size]

    print('#'*80)
    print(online_out[:size])
    print('#'*80)
    print(exact)
    print('#'*80)
    print(online_out[:size] - exact)



def online_convolve(stream1, stream2, out):
    """
    top-down recursive algorithm - useful for analysis but not directly
    applicable to real-time settings
    """
    n = len(stream1) # assumed to be a power of 2
    N = len(out)

    if n == 1:
        out[0] = stream1[0] * stream2[0]
        return out

    middle = n//2
    out = online_convolve(stream1[:middle], stream2[:middle], out)

    end = min(middle + n - 1, N)

    out[middle:end] += convolve(stream1[:middle], stream2[middle:2*middle])[:end-middle]
    out[middle:end] += convolve(stream1[middle:2*middle], stream2[:middle])[:end-middle]

    if n != N:
        out[n:2*n-1] += convolve(stream1[middle:2*middle], stream2[middle:2*middle])[:n-1]

    return out
    







if __name__ == "__main__":
    main()

