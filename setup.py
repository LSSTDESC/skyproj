from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name='skyproj',
    description='Python tools for making sky projections and maps',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/lsstdesc/skyproj',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD-3-Clause',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Astronomy'
    ],
    packages=find_packages(exclude=('tests')),
    package_data={'skyproj': ['data/*.txt', 'data/*.dat']},
    author="Eli Rykoff, Alex Drlica-Wagner",
    author_email='erykoff@stanford.edu',
    install_requires=[
        'numpy',
        'astropy >= 4.0',
        'matplotlib >= 3.1',
        'healpy',
        'healsparse',
        'pyproj >= 3.1'
    ],
    tests_require=[
        'pytest',
        'flake8',
        'pytest-flake8',
        'jupyter',
        'nbconvert'
    ]
)
