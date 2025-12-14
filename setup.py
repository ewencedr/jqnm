"""
Standard setup.py to upload the code on pypi.

    python setup.py sdist bdist_wheel
    twine upload dist/*
"""
import setuptools

with open("README.md", "rb") as fh:
    long_description = fh.read().decode("UTF-8")

import sys
sys.path.append("jqnm")

from _version import __version__

setuptools.setup(
    name="jqnm",
    version=__version__,
    author="Cedric Ewen",
    author_email="",  # Add your email here
    description="JAX-accelerated package for computing Kerr quasinormal mode frequencies with automatic differentiation support",
    keywords='black holes quasinormal modes physics scientific computing numerical methods jax gpu differentiable',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cedricewen/jqnm",  # Update with your GitHub URL
    project_urls={
        "Bug Tracker": "https://github.com/cedricewen/jqnm/issues",
        "Source Code": "https://github.com/cedricewen/jqnm",
        "Original Package": "https://github.com/duetosymmetry/qnm",
    },
    packages=setuptools.find_packages(),
    package_data={'jqnm': ['schwarzschild/data/*', 'LICENSE']},
    python_requires='>=3.9',
    install_requires=[
        'numpy',
        'scipy',
        'jax>=0.4.28',
        'jaxlib>=0.4.28',
        'optimistix>=0.0.6',
        'equinox>=0.11.0',
        'tqdm',
    ],
    extras_require={
        'cuda12': ['jax[cuda12]'],
        'cuda11': ['jax[cuda11]'],
        'test': ['pytest', 'qnm'],
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
    ],
)
