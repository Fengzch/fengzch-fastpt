# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
import os
import torch

def get_cuda_version() -> str:
    cuda_path = os.getenv("CUDA_HOME")
    if not cuda_path:
        raise EnvironmentError("CUDA_HOME environment variable is not set")

    cuda_version_path = os.path.join(cuda_path, "version.txt")
    if not os.path.exists(cuda_version_path):
        raise FileNotFoundError(f"CUDA version file not found: {cuda_version_path}")

    try:
        with open(cuda_version_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
    except IOError as e:
        raise RuntimeError(f"Failed to read CUDA version file: {e}")

    try:
        version_part = content.split()[-1]
        cuda_version = '.'.join(version_part.split('.')[:2])
        return cuda_version
    except (IndexError, AttributeError) as e:
        raise RuntimeError(f"Failed to parse CUDA version from: {content}")

if __name__ == "__main__":
    version_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "torch", "version.py")
    hcu_version = torch.version.__hcu_version__
    version = torch.version.__version__
    debug = torch.version.debug
    git_version = torch.version.git_version
    cuda = get_cuda_version()
    hip = torch.version.hip
    dtk = torch.version.dtk

    with open(version_path, "w") as f:
        f.write("from typing import Optional\n")
        f.write("import os\n\n")
        f.write("USE_FASTPT_CUDA = os.getenv('USE_FASTPT_CUDA', '0') == '1'\n")
        f.write("__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip']\n")
        f.write(f"__version__ = '{version}'\n")
        f.write(f"debug = {debug}\n")
        f.write("if USE_FASTPT_CUDA:\n")
        f.write(f"  cuda: Optional[str] = '{cuda}'\n")
        f.write(f"  hip: Optional[str] = None\n")
        f.write("else:\n")
        f.write(f"  hip: Optional[str] = '{hip}'\n")
        f.write(f"  cuda: Optional[str] = None\n")
        f.write(f"git_version = '{git_version}'\n")
        f.write(f"__hcu_version__ = '{hcu_version}'\n")
        f.write(f"dtk = {dtk}\n")
