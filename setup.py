"""NeuroCUDA — Multi-Backend Neuromorphic Compiler."""
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="neurocuda",
    version="0.2.0",
    description="One-line API for PyTorch → Neuromorphic deployment (GPU, CPU, Loihi 2, FPGA)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Krishna Varma",
    url="https://github.com/neurocuda/neurocuda",
    license="MIT",
    packages=find_packages(include=["neurocuda", "neurocuda.*"]),
    py_modules=["models"],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "all": [
            "snntorch>=0.9.0",
            "nir>=1.0",
            "nirtorch>=2.6",
            "neurobench>=2.3",
            "gymnasium>=1.0",
            "tonic>=1.0",
            "torchvision>=0.15",
        ],
        "nir": ["nir>=1.0", "nirtorch>=2.6"],
        "neurobench": ["neurobench>=2.3"],
        "rl": ["gymnasium>=1.0"],
        "data": ["tonic>=1.0", "torchvision>=0.15"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Hardware",
    ],
    keywords=[
        "neuromorphic", "spiking-neural-networks", "snn", "loihi",
        "fpga", "pytorch", "compiler", "nir", "neurobench",
    ],
)
