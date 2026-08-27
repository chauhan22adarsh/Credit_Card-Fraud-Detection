from setuptools import find_packages, setup

HYPEN_E_DOT = "-e ."


def get_requirements(file_path: str):
    with open(file_path) as f:
        requirements = [line.strip() for line in f.readlines()]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name="credit-card-fraud-detection",
    version="0.0.1",
    author="",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)