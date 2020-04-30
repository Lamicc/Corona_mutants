from setuptools import setup, find_packages

setup(
    name='virusnn',
    version='0.1.0',
    description='virus mutation prediction',
    author='Koptagel, Asplund-Samuelsson, Chen',
    url='https://github.com/Lamicc/Corona_mutants',
    packages=find_packages(include=['virusnn', 'virusnn.*']),
    install_requires=[
        'numpy',
        'scipy',
        'tensorflow',
        'h5py'
    ]
)
