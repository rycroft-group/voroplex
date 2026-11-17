"""
VoroPlex setup script

Builds the Cython/C++ extension, compiling the Voro++ sources into it. Voro++
is included as a Git submodule in voroplex/_lib/voro; clone this repository
with --recursive, or run "git submodule update --init". The package metadata
is in pyproject.toml.
"""

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
import os
import shlex
import subprocess
import sysconfig

# The Voro++ source files, from the Git submodule. These are the files that
# make up the Voro++ library (the "objs" list in voro/src/Makefile).
voro_src_dir = os.path.join("voroplex", "_lib", "voro", "src")
voro_sources = [
    "cell_2d", "cell_3d", "common", "container_2d", "container_3d",
    "container_tri", "iter_2d", "iter_3d", "particle_list", "unitcell",
    "v_base_2d", "v_base_3d", "v_compute_2d", "v_compute_3d", "wall",
    "wall_2d", "wall_3d",
]
if not os.path.exists(os.path.join(voro_src_dir, "voro++.hh")):
    raise RuntimeError(
        f"The Voro++ sources were not found in {voro_src_dir}. They are a Git "
        "submodule: clone with --recursive, or run "
        "'git submodule update --init'."
    )


def read_config_mk(path):
    """Read compiler settings from config.mk, returning a key->value dict."""
    config = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    config[key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    return config


# The compiler is read from config.mk. To change it, edit
# voroplex/_lib/config.mk.
_config = read_config_mk(os.path.join("voroplex", "_lib", "config.mk"))
os.environ["CC"]  = _config.get("cc",  "gcc")
os.environ["CXX"] = _config.get("cxx", "g++")


def openmp_lib_dir(compiler):
    """The directory holding `compiler`'s own OpenMP runtime, or None.

    Asking the compiler is the only reliable way: GCC wants libgomp and
    Clang wants libomp, and either may live outside the default search path.
    """
    for name in ("libgomp.dylib", "libgomp.so.1", "libgomp.so",
                 "libomp.dylib", "libomp.so"):
        try:
            found = subprocess.run(
                [compiler, "-print-file-name=" + name],
                capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
        # The compiler echoes the bare name back when it cannot find the file.
        if found and found != name and os.path.isfile(found):
            return os.path.dirname(os.path.realpath(found))
    return None


# Extensions are linked with LDSHARED from Python's own build configuration
# rather than with CC, so by default the sources are compiled by the compiler
# from config.mk and then linked by whichever compiler built Python. Two
# things go wrong on a Mac with Anaconda, and both end with the wrong OpenMP
# runtime in the extension:
#
#   * Apple's clang does the linking, and its -fopenmp means LLVM's libomp,
#     while the GCC that compiled the objects means libgomp;
#   * Anaconda ships lib/libgomp.dylib as a symlink to its own libomp, and
#     Python's link flags put -L<conda prefix>/lib first, so even GCC's
#     -lgomp is satisfied by LLVM's runtime.
#
# Either way two OpenMP runtimes load into one process -- immediately if
# PyTorch is imported, since it ships libomp -- and the first parallel region
# crashes. So substitute the compiler and put its own OpenMP directory ahead
# of the platform's link flags, which are otherwise left alone. A linker
# command set by the user in the environment is respected as it is.
_LD_WRAPPERS = {"ccache", "sccache", "distcc"}
for _ld_var, _ld_compiler in (("LDSHARED", os.environ["CC"]),
                              ("LDCXXSHARED", os.environ["CXX"])):
    _ld_flags = sysconfig.get_config_var(_ld_var)
    if not _ld_flags or _ld_var in os.environ:
        continue
    _ld_args = shlex.split(_ld_flags)
    # Skip any wrappers, such as ccache, and then the compiler itself.
    while _ld_args and os.path.basename(_ld_args[0]) in _LD_WRAPPERS:
        _ld_args.pop(0)
    _ld_args = _ld_args[1:]
    _omp_dir = openmp_lib_dir(_ld_compiler)
    _prefix = ["-L" + _omp_dir] if _omp_dir else []
    os.environ[_ld_var] = shlex.join([_ld_compiler] + _prefix + _ld_args)


extensions = [
    Extension(
        name="voroplex._lib.voroplusplus_",  # Actual module name (matches .pyx filename)
        sources=[
            "voroplex/_lib/voroplusplus_.pyx",
            "voroplex/_lib/vpp.cpp",
        ] + [os.path.join(voro_src_dir, f + ".cc") for f in voro_sources],
        include_dirs=[
            np.get_include(),
            voro_src_dir,
            "voroplex/_lib",
        ],
        language="c++",
        # VOROPLEX_PROFILE=1 builds a version for benchmarks that times the
        # Voro++ part of the cell loop separately (see vpp.cpp).
        extra_compile_args=["-std=c++11", "-fopenmp", "-O3"]
        + (["-DVOROPLEX_PROFILE"] if os.environ.get("VOROPLEX_PROFILE") else []),
        extra_link_args=["-fopenmp"],
    )
]

# The package metadata (name, version, dependencies, and so on) is in
# pyproject.toml; this file only describes how to build the extension.
setup(
    packages=find_packages(),
    ext_modules=cythonize(extensions, language_level="3"),
    include_package_data=True,  # Include files from MANIFEST.in
    package_data={
        "voroplex._lib": ["config.mk", "*.h", "*.cpp", "*.pyx"],
    },
)
