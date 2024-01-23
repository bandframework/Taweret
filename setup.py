"""
Name: setup.py
Desc: Setup script for Taweret

Start Date: 09/07/23
Version: 1.0
"""

from distutils.core import setup, Extension

setup(
    name='Taweret',
    version='0.1.0',
    description='A Python package for Bayesian model mixing',
    author="Kevin Ingles, Dananjaya (Dan) Liyanage, Alexandra Semposki, John Yannotty",
    author_email='kingles@illinois.edu, liyanage.1@osu.edu, as727414@ohio.edu, yannotty.1@buckeyemail.osu.edu',
    license='MIT',
    url='https://github.com/bandframework/Taweret',
    download_url='',
    keywords=['nuclear physics','model mixing','stacking','gaussian process',"regression trees",'uncertainty quantification'],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        ],
    install_requires=[
        'numpy>=1.20.3',
        'matplotlib',
        'scipy>=1.7.0',
        'seaborn',
        'emcee',
        'corner',
        'scikit-learn',
        'cycler',
        'statistics',
        'bilby',
        'typing',
        'pathlib',
        'ptemcee'
    ]
)
