#!/usr/bin/env python3
"""
Setup script for Earnings Move Predictor.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith('#')
    ]

setup(
    name="earnings-move-predictor",
    version="1.0.0",
    description="Predict stock movements after earnings using options data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Earnings Predictor Team",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "premium": [
            "tradingeconomics>=0.3.0",
            "ib_insync>=0.9.86",
            "polygon-api-client>=1.12.0",
            "alpha_vantage>=2.3.1",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "earnings-predict=src.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    keywords="earnings, options, stocks, trading, finance, prediction",
)
