import setuptools

with open("README.md", 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="knightshock",
    version="0.1.1",
    author="Cory Kinney",
    description="Shock tube experiment data analysis package",
    long_description=long_description,
    url="https://github.com/cory-kinney/KnightShock",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.7"
)
