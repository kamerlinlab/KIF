from setuptools import setup, find_packages

VERSION = "0.3.4"
DESCRIPTION = "Python package for MD simulation analysis"
LONG_DESCRIPTION = """
    A python package to identify the key molecular interactions that regulate any conformational change."""

setup(
    name="KIF",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Rory Crean",
    author_email="rory.crean@kemi.uu.se",
    url="https://github.com/kamerlinlab/KIF",
    packages=find_packages(include=["key_interactions_finder"]),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "scipy",
        "xgboost",
        "catboost",
        "MDAnalysis",
        "MDAnalysisTests",
        "gdown",
        "biopython"
    ],
    package_data={"key_interactions_finder": ["model_params/*.json"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
