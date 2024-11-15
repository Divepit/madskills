from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='madskills',             # Name of your project/library
    version='0.1',                 # Version
    packages=find_packages(),      # Automatically find packages in your project
    install_requires=requirements,
    description='This project implements a library for multi-agent planning environments. It includes a custom grid-based environment, data driven learning algorithms, and benchmarking tools for performance evaluation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Divepit/planning-sandbox-library',  # GitHub repo or homepage
    author='Marco Trentini',
    author_email='marcotr@ethz.ch',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',       # Minimum Python version
)