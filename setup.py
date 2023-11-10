from distutils.core import setup
from setuptools import find_packages
from pathlib import Path

root = Path(__file__).parent

requirements_path = root / "requirements.txt"
with open(requirements_path) as f:
    requirements = [x.strip() for x in f.readlines()]


setup(
    name="translation_llm",
    version="0.0.1",
    packages=find_packages(),
    install_requires=requirements,
)
    