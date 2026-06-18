"""NeuroCUDA — Multi-Backend Neuromorphic Compiler."""
from setuptools import setup, find_packages

setup(
    name="neurocuda",
    version="0.1.0",
    description="One-line API for PyTorch → Neuromorphic deployment",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="NeuroCUDA",
    url="https://github.com/neurocuda/neurocuda",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "snntorch>=0.9.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)