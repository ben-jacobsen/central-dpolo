# Private Continual Counting

This repository contains implementations of the algorithms and experiments described in the paper:

> Ben Jacobsen and Kassem Fawaz, _Private Continual Counting of Unbounded Streams_, 39th Conference on Neural Information Processing Systems (NeurIPS 2025).

## Usage


### Install Dependencies from Pipfile

```bash
pip install --user pipenv
pipenv install
```

### Visualize Approximation Error

Note: The first time these commands are run may be a little slow. By default, the intermediate results that each script computes are cached in a new folder called `seq_cache`, which speeds things up dramatically. If this behavior isn't desired, pass the `--no-cache` argument.

This command generates a plot visualizing the relative error of different orders of asymptotic expansion (Figure 1 in the paper)

```bash
pipenv run python3 singular_approximation.py -T $((2**20)) -K 6  --gamma -0.51 --deltas 0 0.51 --relative
```

### Generate Comparison Plots

This command generates a plot comparing the variance of different private continual counting algorithms as a function of time (Figure 2 in the paper). As above, use `--no-cache` to avoid saving intermediate results.

```bash
pipenv run python3 toeplitz_experiments.py --variance --coefficients --gammas -0.51 --deltas 0 0.51 0.612 -T $((2**24))
```

## Main Components

`sequences.py` - Defines classes representing different strategies for private continual counting.
- `Anytime`: Our algorithm for anytime continual counting
- `Opt`: Near-optimal square-root based matrix factorization for bounded-size inputs, used by Henzinger et al.
- `Independent`: Independent noise at each step (used as subcomponent of the Hybrid mechanism)
- `BinaryMechanism`: The traditional binary tree mechanism of Dwork et al. and Chan et al.
- `SmoothBinary`: A smooth variant of the binary mechanism by Andersson and Pagh
- `DoublingTrick`: A straightforward doubling trick for lifting `Opt` to unbounded inputs
- `Hybrid`: A meta-algorithm by Chan et al. that combines an unbounded mechanism (`Anytime` or `Independent`) with any asymptotically-optimal bounded mechanism (like `Opt`) to create an algorithm that's both unbounded and asymptotically optimal, at the cost of constant factor increase in error. 

`singular_approximation.py` - Contains numerical methods for computing the Taylor coefficients of the functions used by our algorithm.

`toeplitz_experiments` - Main script for generating plots comparing different counting algorithms.
