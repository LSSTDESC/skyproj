from setuptools import setup, find_packages

URL = 'https://github.com/erykoff/skyproj'

with open('requirements.txt') as f:
    install_requires = [req.strip() for req in f.readlines() if req[0] != '#']

setup(
    name='skyproj',
    packages=find_packages(exclude=('tests')),
    # package_data={'skyproj': ['data/*.txt', 'data/*.dat']},
    description="Python tools for making sky maps",
    author="Eli Rykoff, Alex Drlica-Wagner",
    author_email='erykoff@stanford.edu',
    url=URL,
    install_requires=install_requires,
    use_scm_version=True,
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
)
