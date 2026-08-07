# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from .hipify_python import hipify
from .hipify_python import map_extra_args

required_version = '2.12'

try:
    from .cpp_extension import CUDAExtension, CppExtension
    import sys
    from types import ModuleType

    cpp_extensions = ModuleType('cpp_extension')
    cpp_extensions.CUDAExtension = CUDAExtension
    cpp_extensions.CppExtension = CppExtension
    sys.modules[__name__ + '.cpp_extensions'] = cpp_extensions

    import torch
    from packaging import version

    def check_torch_version(required_version):
        current_version = torch.__version__
        current_version_short = ".".join(current_version.split(".")[:2])
        if current_version_short != required_version:
            error_msg = f"Error: The current torch version {current_version_short} does not match the required version {required_version}."
            print(error_msg)
            raise ValueError(error_msg)
        else:
            print(f"Success: The current torch version {current_version_short} matches the required version {required_version}.")

    if required_version:
        check_torch_version(required_version)
except ImportError as e:
    print(f"Warning: CppExtension and CUDAExtension are not available. Just hipify works now.")





__version__ = '2.12.0'

__hcu_version__ = '2.12.0+das.dtk2604'
