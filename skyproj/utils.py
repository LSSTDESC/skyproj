"""
Random utilities
"""
import os
import os.path


def get_datadir():
    from os.path import abspath, dirname, join
    return join(dirname(abspath(__file__)), 'data')


def get_datafile(filename):
    return os.path.join(get_datadir(), filename)
