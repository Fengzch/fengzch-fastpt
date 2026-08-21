# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import glob
import importlib
import importlib.abc
import os
import re
import shlex
import shutil
import setuptools
import subprocess
import sys
import sysconfig
import types
import warnings
import collections
from pathlib import Path
import errno
import logging

logger = logging.getLogger(__name__)
from . import nvcc_args
from . import hipify_python

import torch
import torch._appdirs
from .hipify_python import GeneratedFileCleaner
from .hipify_python import match_nvcc_args, regular_nvcc_args, map_nvcc_args, map_extra_args 
from .nvcc_args import NVCC_ARGS_MAPPING
from .common_flags import COMMON_HIPCC_EXTRA_FLAGS

from typing import Optional, Union
from torch.torch_version import TorchVersion

from setuptools.command.build_ext import build_ext
from pkg_resources import packaging  # type: ignore[attr-defined]

IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform.startswith('darwin')
IS_LINUX = sys.platform.startswith('linux')
LIB_EXT = '.pyd' if IS_WINDOWS else '.so'
EXEC_EXT = '.exe' if IS_WINDOWS else ''
CLIB_PREFIX = '' if IS_WINDOWS else 'lib'
CLIB_EXT = '.dll' if IS_WINDOWS else '.so'
SHARED_FLAG = '/DLL' if IS_WINDOWS else '-shared'

# _HERE = os.path.abspath(__file__)
_HERE = os.path.abspath(torch.__file__)
# _TORCH_PATH = os.path.dirname(os.path.dirname(_HERE))
_TORCH_PATH = os.path.dirname(_HERE)
TORCH_LIB_PATH = os.path.join(_TORCH_PATH, 'lib')


SUBPROCESS_DECODE_ARGS = ('oem',) if IS_WINDOWS else ()
MINIMUM_GCC_VERSION = (5, 0, 0)
MINIMUM_MSVC_VERSION = (19, 0, 24215)

VersionRange = tuple[tuple[int, ...], tuple[int, ...]]
VersionMap = dict[str, VersionRange]
# The following values were taken from the following GitHub gist that
# summarizes the minimum valid major versions of g++/clang++ for each supported
# CUDA version: https://gist.github.com/ax3l/9489132
# Or from include/crt/host_config.h in the CUDA SDK
# The second value is the exclusive(!) upper bound, i.e. min <= version < max
CUDA_GCC_VERSIONS: VersionMap = {
    '11.0': (MINIMUM_GCC_VERSION, (10, 0)),
    '11.1': (MINIMUM_GCC_VERSION, (11, 0)),
    '11.2': (MINIMUM_GCC_VERSION, (11, 0)),
    '11.3': (MINIMUM_GCC_VERSION, (11, 0)),
    '11.4': ((6, 0, 0), (12, 0)),
    '11.5': ((6, 0, 0), (12, 0)),
    '11.6': ((6, 0, 0), (12, 0)),
    '11.7': ((6, 0, 0), (12, 0)),
    '12.0': ((6, 0, 0), (13, 0)),
    '12.1': ((6, 0, 0), (13, 0)),
    '12.2': ((6, 0, 0), (13, 0)),
    '12.3': ((6, 0, 0), (14, 0)),
    '12.4': ((6, 0, 0), (14, 0)),
    '12.5': ((6, 0, 0), (14, 0)),
    '12.6': ((6, 0, 0), (14, 0)),
    '12.7': ((6, 0, 0), (14, 0)),
    '12.8': ((6, 0, 0), (15, 0)),
    '12.9': ((6, 0, 0), (15, 0)),
    '13.0': ((6, 0, 0), (16, 0)),
}

MINIMUM_CLANG_VERSION = (3, 3, 0)
CUDA_CLANG_VERSIONS: VersionMap = {
    '11.1': (MINIMUM_CLANG_VERSION, (11, 0)),
    '11.2': (MINIMUM_CLANG_VERSION, (12, 0)),
    '11.3': (MINIMUM_CLANG_VERSION, (12, 0)),
    '11.4': (MINIMUM_CLANG_VERSION, (13, 0)),
    '11.5': (MINIMUM_CLANG_VERSION, (13, 0)),
    '11.6': (MINIMUM_CLANG_VERSION, (14, 0)),
    '11.7': (MINIMUM_CLANG_VERSION, (14, 0)),
    '12.0': ((7, 0), (15, 0)),
    '12.1': ((7, 0), (15, 0)),
    '12.2': ((7, 0), (16, 0)),
    '12.3': ((7, 0), (16, 0)),
    '12.4': ((7, 0), (17, 0)),
    '12.5': ((7, 0), (18, 0)),
    '12.6': ((7, 0), (18, 0)),
    '12.7': ((7, 0), (19, 0)),
    '12.8': ((7, 0), (19, 0)),
    '12.9': ((7, 0), (19, 0)),
    '13.0': ((7, 0), (21, 0)),
}



def _find_cuda_home() -> str | None:
    """Find the CUDA install path."""
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        nvcc_path = shutil.which("nvcc")
        if nvcc_path is not None:
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        else:
            # Guess #3
            if IS_WINDOWS:
                cuda_homes = glob.glob(
                    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                if len(cuda_homes) == 0:
                    cuda_home = ''
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    if cuda_home and not torch.cuda.is_available():
        logger.warning("No CUDA runtime is found, using CUDA_HOME='%s'", cuda_home)
    return cuda_home

def _find_rocm_home() -> str | None:
    """Find the ROCm install path."""
    # Guess #1
    rocm_home = os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH')
    if rocm_home is None:
        # Guess #2
        hipcc_path = shutil.which('hipcc')
        if hipcc_path is not None:
            rocm_home = os.path.dirname(os.path.dirname(
                os.path.realpath(hipcc_path)))
            # can be either <ROCM_HOME>/hip/bin/hipcc or <ROCM_HOME>/bin/hipcc
            if os.path.basename(rocm_home) == 'hip':
                rocm_home = os.path.dirname(rocm_home)
        else:
            # Guess #3
            fallback_path = '/opt/rocm'
            if os.path.exists(fallback_path):
                rocm_home = fallback_path
    if rocm_home and torch.version.hip is None:
        logger.warning("No ROCm runtime is found, using ROCM_HOME='%s'", rocm_home)
    return rocm_home

def _find_dtk_version(rocm_home):
    version_file = os.path.join(rocm_home, ".dtk_version")
    if not os.path.isfile(version_file):
        return None
    with open(version_file, "r", encoding="utf-8") as file:
        first_line = file.readline().strip()

    match = re.match(
        r"^DTK[-_]([0-9]+)[.]([0-9][0-9])(?:[.]([0-9]+))?(?:[-_]|$)",
        first_line,
    )
    if match is None:
        raise RuntimeError(f"Invalid DTK version: {first_line}")

    patch = int(match.group(3) or 0)
    return int(match.group(1)) * 10000 + int(match.group(2)) * 100 + patch

def _find_sycl_home() -> str | None:
    sycl_home = None
    icpx_path = shutil.which('icpx')
    # Guess 1: for source code build developer/user, we'll have icpx in PATH,
    # which will tell us the SYCL_HOME location.
    if icpx_path is not None:
        sycl_home = os.path.dirname(os.path.dirname(
            os.path.realpath(icpx_path)))

    # Guess 2: for users install Pytorch with XPU support, the sycl runtime is
    # inside intel-sycl-rt, which is automatically installed via pip dependency.
    else:
        try:
            files = importlib.metadata.files('intel-sycl-rt') or []
            for f in files:
                if f.name == "libsycl.so":
                    sycl_home = os.path.dirname(Path(f.locate()).parent.resolve())
                    break
        except importlib.metadata.PackageNotFoundError:
            logger.warning("Trying to find SYCL_HOME from intel-sycl-rt package, but it is not installed.")
    return sycl_home

def _join_rocm_home(*paths) -> str:
    """
    Join paths with ROCM_HOME, or raises an error if it ROCM_HOME is not set.

    This is basically a lazy way of raising an error for missing $ROCM_HOME
    only once we need to get any ROCm-specific path.
    """
    if ROCM_HOME is None:
        raise OSError('ROCM_HOME environment variable is not set. '
                      'Please set it to your ROCm install root.')
    return os.path.join(ROCM_HOME, *paths)

def _join_sycl_home(*paths) -> str:
    """
    Join paths with SYCL_HOME, or raises an error if it SYCL_HOME is not found.

    This is basically a lazy way of raising an error for missing SYCL_HOME
    only once we need to get any SYCL-specific path.
    """
    if SYCL_HOME is None:
        raise OSError('SYCL runtime is not dected. Please setup the pytorch '
                      'prerequisites for Intel GPU following the instruction in '
                      'https://github.com/pytorch/pytorch?tab=readme-ov-file#intel-gpu-support '
                      'or install intel-sycl-rt via pip.')

    return os.path.join(SYCL_HOME, *paths)


ABI_INCOMPATIBILITY_WARNING = '''

                               !! WARNING !!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Your compiler ({}) may be ABI-incompatible with PyTorch!
Please use a compiler that is ABI-compatible with GCC 5.0 and above.
See https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html.

See https://gist.github.com/goldsborough/d466f43e8ffc948ff92de7486c5216d6
for instructions on how to install GCC 5 or higher.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                              !! WARNING !!
'''
WRONG_COMPILER_WARNING = '''

                               !! WARNING !!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Your compiler ({user_compiler}) is not compatible with the compiler Pytorch was
built with for this platform, which is {pytorch_compiler} on {platform}. Please
use {pytorch_compiler} to to compile your extension. Alternatively, you may
compile PyTorch from source using {user_compiler}, and then you can also use
{user_compiler} to compile your extension.

See https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md for help
with compiling PyTorch from source.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                              !! WARNING !!
'''
CUDA_MISMATCH_MESSAGE = '''
The detected CUDA version ({0}) mismatches the version that was used to compile
PyTorch ({1}). Please make sure to use the same CUDA versions.
'''
CUDA_MISMATCH_WARN = "The detected CUDA version ({0}) has a minor version mismatch with the version that was used to compile PyTorch ({1}). Most likely this shouldn't be a problem."
CUDA_NOT_FOUND_MESSAGE = '''
CUDA was not found on the system, please set the CUDA_HOME or the CUDA_PATH
environment variable or add NVCC to your system PATH. The extension compilation will fail.
'''
ROCM_HOME = _find_rocm_home()
HIP_HOME = _join_rocm_home('hip') if ROCM_HOME else None
IS_HIP_EXTENSION = True if ((ROCM_HOME is not None) and (torch.version.hip is not None)) else False

PYTORCH_DTK_VERSION = (
    _find_dtk_version(ROCM_HOME) if IS_HIP_EXTENSION else None
)

ROCM_VERSION = None
if torch.version.hip is not None:
    ROCM_VERSION = tuple(int(v) for v in torch.version.hip.split('.')[:2])

CUDA_HOME = _find_cuda_home()
CUDNN_HOME = os.environ.get('CUDNN_HOME') or os.environ.get('CUDNN_PATH')
SYCL_HOME = _find_sycl_home() if torch.xpu._is_compiled() else None
# PyTorch releases have the version pattern major.minor.patch, whereas when
# PyTorch is built from source, we append the git commit hash, which gives
# it the below pattern.
BUILT_FROM_SOURCE_VERSION_PATTERN = re.compile(r'\d+\.\d+\.\d+\w+\+\w+')


min_supported_cpython = "0x030A0000"  # Python 3.10 hexcode

def get_cxx_compiler():
    if IS_WINDOWS:
        compiler = os.environ.get('CXX', 'cl')
    else:
        compiler = os.environ.get('CXX', 'c++')
    return compiler

def _is_binary_build() -> bool:
    return not BUILT_FROM_SOURCE_VERSION_PATTERN.match(torch.version.__version__)


def _accepted_compilers_for_platform() -> list[str]:
    # gnu-c++ and gnu-cc are the conda gcc compilers
    return ['clang++', 'clang'] if IS_MACOS else ['g++', 'gcc', 'gnu-c++', 'gnu-cc', 'clang++', 'clang']


def get_default_build_root() -> str:
    r'''
    Returns the path to the root folder under which extensions will built.

    For each extension module built, there will be one folder underneath the
    folder returned by this function. For example, if ``p`` is the path
    returned by this function and ``ext`` the name of an extension, the build
    folder for the extension will be ``p/ext``.

    This directory is **user-specific** so that multiple users on the same
    machine won't meet permission issues.
    '''
    return os.path.realpath(torch._appdirs.user_cache_dir(appname='torch_extensions'))


def check_compiler_ok_for_platform(compiler: str) -> bool:
    r'''
    Verifies that the compiler is the expected one for the current platform.

    Args:
        compiler (str): The compiler executable to check.

    Returns:
        True if the compiler is gcc/g++ on Linux or clang/clang++ on macOS,
        and always True for Windows.
    '''
    if IS_WINDOWS:
        return True
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        return False
    ##which = subprocess.check_output(['which', compiler], stderr=subprocess.STDOUT)
    # Use os.path.realpath to resolve any symlinks, in particular from 'c++' to e.g. 'g++'.
    compiler_path = os.path.realpath(compiler_path)
    # Check the compiler name
    if any(name in compiler_path for name in _accepted_compilers_for_platform()):
        return True
    # If compiler wrapper is used try to infer the actual compiler by invoking it with -v flag
    env = os.environ.copy()
    env['LC_ALL'] = 'C'  # Don't localize output
    if compiler == 'hipcc':
        version_string = subprocess.check_output([compiler, '--version'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
    else:
        version_string = subprocess.check_output([compiler, '-v'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
    if IS_LINUX:
        # Check for 'gcc' or 'g++' for sccache wrapper
        pattern = re.compile("^COLLECT_GCC=(.*)$", re.MULTILINE)
        results = re.findall(pattern, version_string)
        if len(results) != 1:
            # Clang is also a supported compiler on Linux
            # Though on Ubuntu it's sometimes called "Ubuntu clang version"
            return 'clang version' in version_string
        compiler_path = os.path.realpath(results[0].strip())
        # On RHEL/CentOS c++ is a gcc compiler wrapper
        if os.path.basename(compiler_path) == 'c++' and 'gcc version' in version_string:
            return True
        return any(name in compiler_path for name in _accepted_compilers_for_platform())
    if IS_MACOS:
        # Check for 'clang' or 'clang++'
        return version_string.startswith("Apple clang")
    return False


def get_compiler_abi_compatibility_and_version(compiler) -> tuple[bool, TorchVersion]:
    """
    Determine if the given compiler is ABI-compatible with PyTorch alongside its version.

    Args:
        compiler (str): The compiler executable name to check (e.g. ``g++``).
            Must be executable in a shell process.

    Returns:
        A tuple that contains a boolean that defines if the compiler is (likely) ABI-incompatible with PyTorch,
        followed by a `TorchVersion` string that contains the compiler version separated by dots.
    """
    if not _is_binary_build():
        return (True, TorchVersion('0.0.0'))
    if os.environ.get('TORCH_DONT_CHECK_COMPILER_ABI') in ['ON', '1', 'YES', 'TRUE', 'Y']:
        return (True, TorchVersion('0.0.0'))

    # First check if the compiler is one of the expected ones for the particular platform.
    if not check_compiler_ok_for_platform(compiler):
        logger.warning(WRONG_COMPILER_WARNING, compiler, _accepted_compilers_for_platform()[0], sys.platform, _accepted_compilers_for_platform()[0])
        return (False, TorchVersion('0.0.0'))

    if IS_MACOS:
        # There is no particular minimum version we need for clang, so we're good here.
        return (True, TorchVersion('0.0.0'))
    try:
        if IS_LINUX:
            minimum_required_version = MINIMUM_GCC_VERSION
            versionstr = subprocess.check_output([compiler, '-dumpfullversion', '-dumpversion'])
            version = versionstr.decode(*SUBPROCESS_DECODE_ARGS).strip().split('.')
        else:
            minimum_required_version = MINIMUM_MSVC_VERSION
            compiler_info = subprocess.check_output(compiler, stderr=subprocess.STDOUT)
            match = re.search(r'(\d+)\.(\d+)\.(\d+)', compiler_info.decode(*SUBPROCESS_DECODE_ARGS).strip())
            version = ['0', '0', '0'] if match is None else list(match.groups())
    except Exception:
        _, error, _ = sys.exc_info()
        logger.warning('Error checking compiler version for %s: %s', compiler, error)
        return (False, TorchVersion('0.0.0'))

    if tuple(map(int, version)) >= minimum_required_version:
        return (True, TorchVersion('.'.join(version)))

    compiler = f'{compiler} {".".join(version)}'
    warnings.warn(ABI_INCOMPATIBILITY_WARNING.format(compiler))

    return (False, TorchVersion('.'.join(version)))

# See below for why we inherit BuildExtension from object.
# https://stackoverflow.com/questions/1713038/super-fails-with-error-typeerror-argument-1-must-be-type-not-classobj-when

def CppExtension(name, sources, *args, **kwargs):
    """
    Create a :class:`setuptools.Extension` for C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a C++ extension.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.

    Example:
        >>> from setuptools import setup
        >>> from torch.utils.cpp_extension import BuildExtension
        >>> from fastpt.cpp_extension import CppExtension
        >>> setup(
        ...     name='extension',
        ...     ext_modules=[
        ...         CppExtension(
        ...             name='extension',
        ...             sources=['extension.cpp'],
        ...             extra_compile_args=['-g']),
        ...     ],
        ...     cmdclass={
        ...         'build_ext': BuildExtension
        ...     })
    """
    if _check_cppextension_to_cu(name, sources, *args, **kwargs):
        return CUDAExtension(name, sources, *args, **kwargs)
    include_dirs = kwargs.get('include_dirs', [])
    include_dirs += include_paths()
    kwargs['include_dirs'] = include_dirs

    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += library_paths()
    kwargs['library_dirs'] = library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('c10')
    libraries.append('torch')
    libraries.append('torch_cpu')
    if not kwargs.get('py_limited_api', False):
        # torch_python uses more than the python limited api
        libraries.append('torch_python')
    if IS_WINDOWS:
        libraries.append("sleef")

    kwargs['libraries'] = libraries

    kwargs['language'] = 'c++'
    return setuptools.Extension(name, sources, *args, **kwargs)

def CUDAExtension(name, sources, *args, **kwargs):
    r'''
    Creates a :class:`setuptools.Extension` for CUDA/C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a CUDA/C++
    extension. This includes the CUDA include path, library path and runtime
    library.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.

    Example:
        >>> from setuptools import setup
        >>> from torch.utils.cpp_extension import BuildExtension
        >>> from fastpt.cpp_extension import CUDAExtension
        >>> setup(
        ...     name='cuda_extension',
        ...     ext_modules=[
        ...         CUDAExtension(
        ...                 name='cuda_extension',
        ...                 sources=['extension.cpp', 'extension_kernel.cu'],
        ...                 extra_compile_args={'cxx': ['-g'],
        ...                                     'nvcc': ['-O2']})
        ...     ],
        ...     cmdclass={
        ...         'build_ext': BuildExtension
        ...     })

    Compute capabilities:

    By default the extension will be compiled to run on all archs of the cards visible during the
    building process of the extension, plus PTX. If down the road a new card is installed the
    extension may need to be recompiled. If a visible card has a compute capability (CC) that's
    newer than the newest version for which your nvcc can build fully-compiled binaries, PyTorch
    will make nvcc fall back to building kernels with the newest version of PTX your nvcc does
    support (see below for details on PTX).

    You can override the default behavior using `TORCH_CUDA_ARCH_LIST` to explicitly specify which
    CCs you want the extension to support:

    TORCH_CUDA_ARCH_LIST="6.1 8.6" python build_my_extension.py
    TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX" python build_my_extension.py

    The +PTX option causes extension kernel binaries to include PTX instructions for the specified
    CC. PTX is an intermediate representation that allows kernels to runtime-compile for any CC >=
    the specified CC (for example, 8.6+PTX generates PTX that can runtime-compile for any GPU with
    CC >= 8.6). This improves your binary's forward compatibility. However, relying on older PTX to
    provide forward compat by runtime-compiling for newer CCs can modestly reduce performance on
    those newer CCs. If you know exact CC(s) of the GPUs you want to target, you're always better
    off specifying them individually. For example, if you want your extension to run on 8.0 and 8.6,
    "8.0+PTX" would work functionally because it includes PTX that can runtime-compile for 8.6, but
    "8.0 8.6" would be better.

    Note that while it's possible to include all supported archs, the more archs get included the
    slower the building process will be, as it will build a separate kernel image for each arch.

    Note that CUDA-11.5 nvcc will hit internal compiler error while parsing torch/extension.h on Windows.
    To workaround the issue, move python binding logic to pure C++ file.

    Example use:
        #include <ATen/ATen.h>
        at::Tensor SigmoidAlphaBlendForwardCuda(....)

    Instead of:
        #include <torch/extension.h>
        torch::Tensor SigmoidAlphaBlendForwardCuda(...)

    Currently open issue for nvcc bug: https://github.com/pytorch/pytorch/issues/69460
    Complete workaround code example: https://github.com/facebookresearch/pytorch3d/commit/cb170ac024a949f1f9614ffe6af1c38d972f7d48

    Relocatable device code linking:

    If you want to reference device symbols across compilation units (across object files),
    the object files need to be built with `relocatable device code` (-rdc=true or -dc).
    An exception to this rule is "dynamic parallelism" (nested kernel launches)  which is not used a lot anymore.
    `Relocatable device code` is less optimized so it needs to be used only on object files that need it.
    Using `-dlto` (Device Link Time Optimization) at the device code compilation step and `dlink` step
    helps reduce the protentional perf degradation of `-rdc`.
    Note that it needs to be used at both steps to be useful.

    If you have `rdc` objects you need to have an extra `-dlink` (device linking) step before the CPU symbol linking step.
    There is also a case where `-dlink` is used without `-rdc`:
    when an extension is linked against a static lib containing rdc-compiled objects
    like the [NVSHMEM library](https://developer.nvidia.com/nvshmem).

    Note: Ninja is required to build a CUDA Extension with RDC linking.

    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)
        >>> CUDAExtension(
        ...        name='cuda_extension',
        ...        sources=['extension.cpp', 'extension_kernel.cu'],
        ...        dlink=True,
        ...        dlink_libraries=["dlink_lib"],
        ...        extra_compile_args={'cxx': ['-g'],
        ...                            'nvcc': ['-O2', '-rdc=true']})
    '''
    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += library_paths(device_type="cuda")
    kwargs['library_dirs'] = library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('c10')
    libraries.append('torch')
    libraries.append('torch_cpu')
    if not kwargs.get('py_limited_api', False):
        # torch_python uses more than the python limited api
        libraries.append('torch_python')
    if IS_HIP_EXTENSION:
        libraries.append('amdhip64')
        libraries.append('c10_hip')
        libraries.append('torch_hip')
    else:
        libraries.append('cudart')
        libraries.append('c10_cuda')
        libraries.append('torch_cuda')
    kwargs['libraries'] = libraries

    include_dirs = kwargs.get('include_dirs', [])

    if IS_HIP_EXTENSION:
        build_dir = os.getcwd()
        ignores =  kwargs.get('ignores', [])
        hipify_result = hipify_python.hipify(
            project_directory=build_dir,
            output_directory=build_dir,
            header_include_dirs=include_dirs,
            includes=[os.path.join(build_dir, '*')],  # limit scope to build_dir only
            extra_files=[os.path.abspath(s) for s in sources],
            show_detailed=True,
            ignores = ignores,
            is_pytorch_extension=True,
            hipify_extra_files_only=True,  # don't hipify everything in includes path
            add_dtk_macros = True, # 在torch2.1以及以上版本，进行头文件处理
        )

        hipified_sources = set()
        for source in sources:
            s_abs = os.path.abspath(source)
            hipified_s_abs = (hipify_result[s_abs].hipified_path if (s_abs in hipify_result and
                              hipify_result[s_abs].hipified_path is not None) else s_abs)
            # setup() arguments must *always* be /-separated paths relative to the setup.py directory,
            # *never* absolute paths
            hipified_sources.add(os.path.relpath(hipified_s_abs, build_dir))

        sources = list(hipified_sources)

    include_dirs += include_paths(device_type="cuda")
    kwargs['include_dirs'] = include_dirs

    kwargs['language'] = 'c++'

    dlink_libraries = kwargs.get('dlink_libraries', [])
    dlink = kwargs.get('dlink', False) or dlink_libraries
    if dlink:
        extra_compile_args = kwargs.get('extra_compile_args', {})

        extra_compile_args_dlink = extra_compile_args.get('nvcc_dlink', [])
        extra_compile_args_dlink += ['-dlink']
        extra_compile_args_dlink += [f'-L{x}' for x in library_dirs]
        extra_compile_args_dlink += [f'-l{x}' for x in dlink_libraries]

        if (torch.version.cuda is not None) and TorchVersion(torch.version.cuda) >= '11.2':
            extra_compile_args_dlink += ['-dlto']   # Device Link Time Optimization started from cuda 11.2

        extra_compile_args['nvcc_dlink'] = extra_compile_args_dlink

        kwargs['extra_compile_args'] = extra_compile_args
    #aiss add for nvcc args in cudaExtension
    extra_compile_args = kwargs.get('extra_compile_args', {})
    output_nvcc_argslist = copy.deepcopy(COMMON_HIPCC_EXTRA_FLAGS)

    if PYTORCH_DTK_VERSION is not None:
        output_nvcc_argslist.append(
            f"-DPYTORCH_DTK_VERSION={PYTORCH_DTK_VERSION}"
        )

    if type(extra_compile_args).__name__ != 'list':# if pextra_compile_args is list,get args from extra_compile_args will failed with error
        extra_nvcc_args = extra_compile_args.get('nvcc', [])
        extra_cxx_args = extra_compile_args.get('cxx', [])# cuda mix hip add ,
    
        if(len(extra_compile_args)==0):
            kwargs['extra_compile_args'] = {}
        if(extra_nvcc_args):
            output_nvcc_argslist = map_nvcc_args(extra_nvcc_args) + output_nvcc_argslist
        kwargs['extra_compile_args']['nvcc'] = output_nvcc_argslist
        if(extra_cxx_args):
            kwargs['extra_compile_args']['cxx'] = map_nvcc_args(extra_cxx_args)
    elif isinstance(extra_compile_args, list):
        flagIn = kwargs['extra_compile_args']
        output_nvcc_argslist = output_nvcc_argslist+flagIn
        kwargs['extra_compile_args'] = {}
        kwargs['extra_compile_args']['nvcc'] = map_nvcc_args(output_nvcc_argslist)
        kwargs['extra_compile_args']['cxx'] = flagIn
        
#end for nvcc preprocess
    return setuptools.Extension(name, sources, *args, **kwargs)

def include_paths(device_type: str = "cpu", torch_include_dirs=True) -> list[str]:
    """
    Get the include paths required to build a C++ or CUDA or SYCL extension.

    Args:
        device_type: Defaults to "cpu".

    Returns:
        A list of include path strings.
    """
    paths = []
    lib_include = os.path.join(_TORCH_PATH, 'include')
    if torch_include_dirs:
        paths.extend([
            lib_include,
            # Remove this once torch/torch.h is officially no longer supported for C++ extensions.
            os.path.join(lib_include, 'torch', 'csrc', 'api', 'include'),
        ])
    if device_type == "cuda" and IS_HIP_EXTENSION:
        paths.append(os.path.join(lib_include, 'THH'))
        paths.append(_join_rocm_home('include'))
    elif device_type == "cuda":
        cuda_home_include = _join_cuda_home('include')
        # if we have the Debian/Ubuntu packages for cuda, we get /usr as cuda home.
        # but gcc doesn't like having /usr/include passed explicitly
        if cuda_home_include != '/usr/include':
            paths.append(cuda_home_include)

        # Support CUDA_INC_PATH env variable supported by CMake files
        if (cuda_inc_path := os.environ.get("CUDA_INC_PATH", None)) and \
                cuda_inc_path != '/usr/include':
            paths.append(cuda_inc_path)
        if CUDNN_HOME is not None:
            paths.append(os.path.join(CUDNN_HOME, 'include'))
    elif device_type == "xpu":
        paths.append(_join_sycl_home('include'))
        paths.append(_join_sycl_home('include', 'sycl'))
    return paths


def library_paths(device_type: str = "cpu", torch_include_dirs: bool = True, cross_target_platform: str | None = None) -> list[str]:
    """
    Get the library paths required to build a C++ or CUDA extension.

    Args:
        device_type: Defaults to "cpu".

    Returns:
        A list of library path strings.
    """

    paths = []

    if torch_include_dirs:
        # We need to link against libtorch.so
        paths.extend([TORCH_LIB_PATH])

    if device_type == "cuda" and IS_HIP_EXTENSION:
        lib_dir = 'lib'
        paths.append(_join_rocm_home(lib_dir))
        if HIP_HOME is not None:
            paths.append(os.path.join(HIP_HOME, 'lib'))
    elif device_type == "cuda":
        if cross_target_platform == "windows":
            lib_dir = os.path.join('lib', 'x64')
            if WINDOWS_CUDA_HOME is None:
                raise RuntimeError("Need to set WINDOWS_CUDA_HOME for windows cross-compilation")
            paths.append(os.path.join(WINDOWS_CUDA_HOME, lib_dir))
        else:
            if IS_WINDOWS:
                lib_dir = os.path.join('lib', 'x64')
            else:
                lib_dir = 'lib64'
                if (not os.path.exists(_join_cuda_home(lib_dir)) and
                        os.path.exists(_join_cuda_home('lib'))):
                    # 64-bit CUDA may be installed in 'lib' (see e.g. gh-16955)
                    # Note that it's also possible both don't exist (see
                    # _find_cuda_home) - in that case we stay with 'lib64'.
                    lib_dir = 'lib'

        paths.append(_join_cuda_home(lib_dir))
        if CUDNN_HOME is not None:
            paths.append(os.path.join(CUDNN_HOME, lib_dir))
    elif device_type == "xpu":
        if IS_WINDOWS:
            lib_dir = os.path.join('lib', 'x64')
        else:
            lib_dir = 'lib64'
            if (not os.path.exists(_join_sycl_home(lib_dir)) and
                    os.path.exists(_join_sycl_home('lib'))):
                lib_dir = 'lib'

        paths.append(_join_sycl_home(lib_dir))

    return paths

def check_compiler_is_gcc(compiler):
    if not IS_LINUX:
        return False

    env = os.environ.copy()
    env['LC_ALL'] = 'C'  # Don't localize output
    version_string = subprocess.check_output([compiler, '-v'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
    # Check for 'gcc' or 'g++' for sccache wrapper
    pattern = re.compile("^COLLECT_GCC=(.*)$", re.MULTILINE)
    results = re.findall(pattern, version_string)
    if len(results) != 1:
        return False
    compiler_path = os.path.realpath(results[0].strip())
    # On RHEL/CentOS c++ is a gcc compiler wrapper
    if os.path.basename(compiler_path) == 'c++' and 'gcc version' in version_string:
        return True
    return False

def remove_extension_h_precompiler_headers():
    def _remove_if_file_exists(path_file):
        if os.path.exists(path_file):
            os.remove(path_file)

    head_file_pch = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h.gch')
    head_file_signature = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h.sign')

    _remove_if_file_exists(head_file_pch)
    _remove_if_file_exists(head_file_signature)


def _join_cuda_home(*paths) -> str:
    r'''
    Joins paths with CUDA_HOME, or raises an error if it CUDA_HOME is not set.

    This is basically a lazy way of raising an error for missing $CUDA_HOME
    only once we need to get any CUDA-specific path.
    '''
    if CUDA_HOME is None:
        raise OSError('CUDA_HOME environment variable is not set. '
                      'Please set it to your CUDA install root.')
    return os.path.join(CUDA_HOME, *paths)

def _is_cuda_file(path: str) -> bool:
    valid_ext = ['.cu', '.cuh']
    if IS_HIP_EXTENSION:
        valid_ext.append('.hip')
    return os.path.splitext(path)[1] in valid_ext

def _check_cppextension_to_cu(name, sources, *args, **kwargs):
    """
    check cppextension, if it has nvcc compile args,change the CPPExtension to CUDAExtension
    """
    extra_compile_args = kwargs.get('extra_compile_args', {})
    if isinstance(extra_compile_args, dict):
        if('nvcc' in extra_compile_args):
            return True
