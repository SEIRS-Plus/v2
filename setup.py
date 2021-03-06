# Standard Libraries
from setuptools import setup, find_namespace_packages
import pathlib


APP_NAME = "seirsplus"
VERSION = "2.0.0"
LICENSE = "MIT"
AUTHOR = "Ryan Seamus McGee"
DESCRIPTION = "Framework for modeling epidemic dynamics with interventions."
URL = "https://github.com/SEIRS-Plus/v2.git"


# Directory containing this file
HERE = pathlib.Path(__file__).parent

# Text of README file
README = (HERE / "README.md").read_text()


setup(
    name=APP_NAME,
    version=VERSION,
    license=LICENSE,
    author=AUTHOR,
    description=DESCRIPTION,
    long_description=README,
    long_description_content_type="text/markdown",
    url=URL,
    install_requires=[
        "networkx>=2.0",
        "numpy",
        "scipy",
        "matplotlib",
    ],
    extra_require={
        "docs": ["Sphinx>=4.2.0"],
        "notebooks": ["jupyterlab>=3.2.0", "altair>=4.1.0"],
        "dev": ["pre-commit>=2.15.0"],
    },
    packages=find_namespace_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["*.yaml"]},
    entry_points="""
    [console_scripts]
    """,
    python_requires=">=2.6",
    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True
)
