"""
Test notebooks in tutorial.

Adapted from:
https://blog.thedataincubator.com/2016/06/testing-jupyter-notebooks/
"""
import os
import pytest
import subprocess
import tempfile

import nbformat


ROOT = os.path.abspath(os.path.dirname(__file__))


def _notebook_run(nbfile):
    """Execute a notebook via nbconvert and collect output.

    Parameters
    ----------
    nbfile : `str`
        Notebook file to run.

    Returns
    -------
    nb : `???`
        Parsed notebook object.
    errors : `list` [`str`]
        List of error strings.
    """
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=60", "--log-level=WARN",
                "--output", fout.name, nbfile]
        subprocess.check_call(args)

        fout.seek(0)
        nb = nbformat.read(fout, nbformat.current_nbformat)

    errors = [output for cell in nb.cells if "outputs" in cell
              for output in cell["outputs"]
              if output.output_type == "error"]

    return nb, errors


@pytest.mark.parametrize("nbfile", ["tutorial_baseclass.ipynb",
                                    "tutorial_surveys.ipynb",
                                    "tutorial_healsparse.ipynb"])
def test_tutorial_notebooks(nbfile):
    """Test running a tutorial notebook."""
    fname = os.path.abspath(os.path.join(ROOT, '../tutorial', nbfile))
    nb, errors = _notebook_run(fname)
    assert errors == []
