from setuptools import setup, find_packages
import re
import os

VERSION = '0.1.0'
here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='gpsmatcher',
    version=VERSION,
    packages=find_packages(),
    long_description=long_description,
    author='Bonnetain Loïc',
    author_email='loic.bonnetain@univ-eiffel.fr',
    url='https://github.com/lbonnetain/gpsmatcher',
    description='Match a trace of GPS positions to a transportation graph',
    python_requires='>=3.6',
    license='GPLv3+',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    keywords='map matching',
)