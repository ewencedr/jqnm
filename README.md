# jqnm - JAX-Accelerated Kerr Quasinormal Modes

[![PyPI version](https://badge.fury.io/py/jqnm.svg)](https://badge.fury.io/py/jqnm)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`jqnm` is a JAX-accelerated Python package for computing Kerr black hole
quasinormal mode frequencies, angular separation constants, and
spherical-spheroidal mixing coefficients.

This package is a **JAX-based rewrite** of the excellent [qnm](https://github.com/duetosymmetry/qnm) 
package by [Leo C. Stein](https://duetosymmetry.com), providing:

- **GPU acceleration** via JAX
- **Automatic differentiation** - compute gradients of QNM frequencies with respect to spin
- **JIT compilation** for faster repeated computations
- Full compatibility with the original `qnm` data cache

## Installation

### PyPI

```shell
pip install jqnm
```

### From source

```shell
git clone https://github.com/cedricewen/jqnm.git
cd jqnm
pip install .
```

### GPU Support

For GPU acceleration, install JAX with CUDA support:

```shell
# For CUDA 12
pip install jqnm[cuda12]

# For CUDA 11  
pip install jqnm[cuda11]
```

## Dependencies

- numpy
- scipy
- [JAX](https://github.com/google/jax) (>=0.4.28)
- [optimistix](https://github.com/patrick-kidger/optimistix) - differentiable root finding
- [equinox](https://github.com/patrick-kidger/equinox)
- tqdm (for download progress)

## Quick Start

```python
import jqnm

# Download precomputed mode data (only need to do this once)
jqnm.download_data()

# Get the (2,2,0) gravitational mode
grav_220 = jqnm.modes_cache(s=-2, l=2, m=2, n=0)

# Compute QNM frequency at spin a=0.68
omega, A, C = grav_220(a=0.68)
print(omega)
# (0.5239751042900845-0.08151262363119986j)
```

## Automatic Differentiation

One of the key features of `jqnm` is the ability to compute gradients:

```python
import jax
import jax.numpy as jnp
import jqnm

# Get a mode
mode = jqnm.modes_cache(s=-2, l=2, m=2, n=0)

# Define a function that returns the real part of omega
def get_omega_real(a):
    omega, A, C = mode(a=a)
    return jnp.real(omega)

# Compute the derivative with respect to spin
d_omega_da = jax.grad(get_omega_real)
print(d_omega_da(0.5))
```

## Usage

The highest-level interface is via `jqnm.modes_cache`, which loads cached 
*spin sequences* from disk. A spin sequence is a mode labeled by (s,l,m,n), 
with the spin `a` ranging from a=0 to some maximum (e.g., 0.9995).

```python
import jqnm
import numpy as np
import matplotlib.pyplot as plt

# Get multiple overtones
s, l, m = (-2, 2, 2)
mode_list = [(s, l, m, n) for n in np.arange(0, 7)]
modes = {ind: jqnm.modes_cache(*ind) for ind in mode_list}

# Plot omega trajectories
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
for mode, seq in modes.items():
    plt.plot(np.real(seq.omega), np.imag(seq.omega))
plt.xlabel('Re(ω)')
plt.ylabel('Im(ω)')
plt.title('QNM Frequencies')

plt.subplot(1, 2, 2)
for mode, seq in modes.items():
    plt.plot(np.real(seq.A), np.imag(seq.A))
plt.xlabel('Re(A)')
plt.ylabel('Im(A)')
plt.title('Separation Constants')
plt.tight_layout()
plt.show()
```

## Spherical-Spheroidal Decomposition

The angular dependence of QNMs are spin-weighted *spheroidal* harmonics. 
The package returns coefficients C for expressing spheroidals in terms of 
sphericals:

```python
import jqnm

grav_220 = jqnm.modes_cache(s=-2, l=2, m=2, n=0)
omega, A, C = grav_220(a=0.68)

# Get the corresponding l values
ells = jqnm.angular.ells(s=-2, m=2, l_max=grav_220.l_max)
print(f"ell values: {ells}")
print(f"C coefficients: {C}")
```

## Credits and Acknowledgments

This package is a JAX-based rewrite of the [qnm](https://github.com/duetosymmetry/qnm) 
package created by [Leo C. Stein](https://duetosymmetry.com). The original package 
and its scientific foundations are described in:

> Stein, Leo C. (2019). *qnm: A Python package for calculating Kerr quasinormal 
> modes, separation constants, and spherical-spheroidal mixing coefficients.* 
> Journal of Open Source Software, 4(42), 1683. 
> [doi:10.21105/joss.01683](https://doi.org/10.21105/joss.01683)
> [arXiv:1908.10377](https://arxiv.org/abs/1908.10377)

If you use this package in academic work, please cite the original `qnm` paper above.

## License

MIT License - see [LICENSE](LICENSE) for details.

This package is derived from [qnm](https://github.com/duetosymmetry/qnm) 
(Copyright © 2019 Leo C. Stein), which is also MIT licensed.
