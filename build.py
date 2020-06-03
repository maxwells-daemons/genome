"""
Script to build the project code with a Poetry hook.

Based on:
    - https://github.com/ADicksonLab/geomm/blob/
      d5ba96b2af6622c7eb87a64f719591bcaa530f57/build.py.
      (Poetry build hooks)
    - https://github.com/rmcgibbo/npcuda-example/blob/master/cython/setup.py
      (Building Cuda code as part of the Cython build process)
"""
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import os

# Adapted fom http://code.activestate.com/recipes/52224
def find_in_path(name, path):
    """Find a file in a search path"""

    for dir in path.split(os.pathsep):
        binpath = os.path.join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


# Taken from: https://github.com/rmcgibbo/npcuda-example/blob/master/cython/setup.py
def locate_cuda():
    """
    Locate the CUDA environment on the system.
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDA_HOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """

    # First check if the CUDA_HOME env variable is in use
    if "CUDA_HOME" in os.environ:
        home = os.environ["CUDA_HOME"]
        nvcc = os.path.join(home, "bin", "nvcc")
    else:
        # Otherwise, search the PATH for NVCC
        nvcc = find_in_path("nvcc", os.environ["PATH"])
        if nvcc is None:
            raise EnvironmentError(
                "The nvcc binary could not be "
                "located in your $PATH. Either add it to your path, "
                "or set $CUDA_HOME"
            )
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {
        "home": home,
        "nvcc": nvcc,
        "include": os.path.join(home, "include"),
        "lib64": os.path.join(home, "lib64"),
    }
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError(
                "The CUDA %s path could not be " "located in %s" % (k, v)
            )

    return cudaconfig


CUDA = locate_cuda()


# Taken from: https://github.com/rmcgibbo/npcuda-example/blob/master/cython/setup.py
def customize_compiler_for_nvcc(self):
    """
    Inject deep into distutils to customize how the dispatch to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kind of like a wierd functional
    subclassing going on.
    """

    # Tell the compiler it can processes .cu
    self.src_extensions.append(".cu")

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == ".cu":
            # use the cuda for .cu files
            self.set_executable("compiler_so", CUDA["nvcc"])
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs["nvcc"]
        else:
            postargs = extra_postargs["gcc"]

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile


# Customize the compiler extension to build .cu files with NVCC
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


# An extension for CPU inference code
cpu_extension = Extension(
    "cpu_inference",
    sources=["genome/binary_networks/cpu.pyx"],
    library_dirs=[],
    libraries=[],
    runtime_library_dirs=[],
    include_dirs=[np.get_include()],
    language="c",
    extra_compile_args={"gcc": [], "nvcc": []},  # type: ignore
)

# An extension for GPU inference code
gpu_extension = Extension(
    "gpu_inference",
    sources=[
        "genome/binary_networks/gpu/python_interface.pyx",
        "genome/binary_networks/gpu/binary_network.cpp",
        "genome/binary_networks/gpu/sign_layer.cu",
        "genome/binary_networks/gpu/linear_layer.cu",
    ],
    library_dirs=[CUDA["lib64"]],
    libraries=["cudart"],
    runtime_library_dirs=[CUDA["lib64"]],
    include_dirs=[np.get_include(), CUDA["include"]],
    language="c++",
    extra_compile_args={  # type: ignore
        "gcc": [],
        "nvcc": [
            "-arch=sm_30",
            "--ptxas-options=-v",
            "-c",
            "--compiler-options",
            "'-fPIC'",
        ],
    },
)

extensions = [cpu_extension, gpu_extension]


def build(setup_kwargs):
    """
    Build the Cython code used for this project.

    NOTE: this is undocumented behavior of Poetry and subject to change.
    See: https://github.com/python-poetry/poetry/issues/11.
    """
    setup_kwargs.update(
        {
            "ext_modules": extensions,
            "include_dirs": [np.get_include()],
            "cmdclass": dict(build_ext=custom_build_ext),
        }
    )
