from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "Python package for MD simulation analysis"
LONG_DESCRIPTION = """
    A python package to identify the key molecular interactions that regulate any conformational change."""

setup(name="KIF",
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author="Rory Crean",
      url="https://github.com/kamerlinlab/KIF",
      packages=find_packages(
          include=["key_interactions_finder"]),
      install_requires=[
          "pandas",
          "numpy",
          "sklearn",
          "scipy",
          "xgboost",
          "catboost",
          "MDAnalysis",
          "gdown",
      ],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: GNU General Public License v2.0",
          "Operating System :: OS Independent",
      ])
