"""
Script to build the project code with a Poetry hook.

See: https://github.com/ADicksonLab/geomm/blob/
     d5ba96b2af6622c7eb87a64f719591bcaa530f57/build.py.
"""

import Cython.Build
import numpy as np

modules = ["genome/binary_layers.pyx"]
extensions = Cython.Build.cythonize(
    modules,
    compiler_directives={
        "boundscheck": False,
        "wraparound": False,
        "initializedcheck": False,
        "language_level": 3,
    },
)


def build(setup_kwargs):
    """
    Build the Cython code used for this project.

    NOTE: this is undocumented behavior of Poetry and subject to change.
    See: https://github.com/python-poetry/poetry/issues/11.
    """

    setup_kwargs.update({"ext_modules": extensions, "include_dirs": [np.get_include()]})
