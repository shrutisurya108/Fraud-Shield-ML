"""
fraud-shield-ml | setup.py
Allows: pip install -e .
Makes src/ importable as a package anywhere in the project.
"""

from setuptools import setup, find_packages

setup(
    name="fraud_shield_ml",
    version="1.0.0",
    description="End-to-end credit card fraud detection — IEEE-CIS dataset",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.5.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.3.0",
        "imbalanced-learn>=0.12.0",
        "shap>=0.45.0",
        "lime>=0.2.0.1",
        "optuna>=3.6.0",
        "loguru>=0.7.0",
        "pyyaml>=6.0.1",
        "streamlit>=1.35.0",
    ],
)
