# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

try:
    import torch
except ImportError:
    raise ImportError("Torch is required")
import os
import shutil
import subprocess
import sys
import platform

include_ends = ['.h', '.cuh', '.hpp', '.py']
required_version='2.12'


def check_torch_version(required_version):
    current_version = torch.__version__
    current_version = ".".join(current_version.split(".")[:2]) 
    if current_version != required_version:
        error_msg = f"Torch version mismatch. Required: {required_version}, Found: {current_version}"
        raise RuntimeError(error_msg)
    
    print(f"Torch version is correct: {current_version}")

distro_full = platform.freedesktop_os_release().get("PRETTY_NAME", "Unknown")
distro_name = distro_full.split()[0] if distro_full != "Unknown" else "Unknown"
ROCM_PATH = os.getenv('ROCM_PATH', "/opt/dtk/")
dtk_path = ROCM_PATH + '/.info/rocm_version'
with open(dtk_path, 'r') as file:
    content = file.read().strip()
dtk_version = content.replace('.', '')

def cuda_init():
    try:
        check_torch_version(required_version)
        print("torch version check success!")
    except RuntimeError as e:
        print(e)
        sys.exit(1)
    if os.getenv("FASTPT_DEBUG", "0") == "1":
        print("Custom installing...")
    torch_install_path_ = os.path.dirname(torch.__file__)
    fastpt_install_path_ = os.path.dirname(__file__)

    target_dir = torch_install_path_
    source_dir = os.path.join(fastpt_install_path_, './torch')  # 源文件夹
    py_files = [
            (os.path.join(source_dir, 'version.py'), os.path.join(torch_install_path_, 'version.py')),
            (os.path.join(source_dir,'cuda/__init__.py'), os.path.join(torch_install_path_, 'cuda', '__init__.py')),
            (os.path.join(source_dir,'utils/cpp_extension.py'), os.path.join(torch_install_path_, 'utils', 'cpp_extension.py')),
            (os.path.join(source_dir,'share/cmake/Caffe2/Caffe2Targets.cmake'),  os.path.join(torch_install_path_,'share/cmake/Caffe2', 'Caffe2Targets.cmake')),
            (os.path.join(source_dir,'share/cmake/Caffe2/Caffe2Targets-release.cmake'), os.path.join(torch_install_path_,'share/cmake/Caffe2', 'Caffe2Targets-release.cmake')),
        ]
    for src, dst in py_files:
        if os.path.exists(src):
            dst_dir = os.path.dirname(dst)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(src, dst)
            if os.getenv("FASTPT_DEBUG", "0") == "1":
                print(f'Successfully installed {src} -> {dst}')

    if os.path.exists(source_dir):
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if any([file.endswith(file_end) for file_end in include_ends + ['.so']]):
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(target_dir, os.path.relpath(src_file, source_dir))
                    dst_dir  = os.path.dirname(dst_file)
                    
                    os.makedirs(dst_dir, exist_ok=True)
                    if os.getenv("FASTPT_DEBUG", "0") == "1":
                        print(f'{dst_file} is changed')

                    shutil.copy(src_file, dst_file)
    else:
        print(f"source_dir {source_dir} not found")
    
    torch_lib_path = os.path.join(torch_install_path_, "lib")
    cuda_lib_path = os.getenv("ROCM_PATH", "/opt/dtk/")
    if dtk_version == '2504':
        cuda_lib_path_str  = os.path.join(cuda_lib_path, "cuda/lib64/")
    elif dtk_version == '25041':
        cuda_lib_path_str  = os.path.join(cuda_lib_path, "cuda/cuda-11/lib64/")
    else :
        cuda_lib_path_str  = os.path.join(cuda_lib_path, "cuda/cuda-12/lib64/")
    fastpt_torch_lib_path_str = f"FASTPT_TORCH_LIB_PATH={torch_lib_path}"
    fastpt_cuda_lib_path_str = f"FASTPT_CUDA_LIB_PATH={cuda_lib_path_str}"

    torch_path_env_str = f"export LD_LIBRARY_PATH=$FASTPT_TORCH_LIB_PATH:$LD_LIBRARY_PATH"
    cuda_path_env_str = f"export LD_LIBRARY_PATH=$FASTPT_CUDA_LIB_PATH:$LD_LIBRARY_PATH"

    if os.getenv("FASTPT_DEBUG", "0") == "1":
        print(f"torch_lib_path    : {torch_lib_path}")
        print(f"cuda_lib_path     : {cuda_lib_path}")
        print(f"torch_path_env_str: {torch_path_env_str}")
        print(f"cuda_path_env_str : {cuda_path_env_str}")

    bashrc_path = os.path.expanduser("~/.bashrc") if distro_name != "Kylin" else os.path.abspath("/etc/profile")
    with open(bashrc_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    fastpt_torch_found = False
    fastpt_cuda_found = False
    torch_env_found = False
    cuda_env_found = False

    for line in lines:
        if line.startswith("FASTPT_TORCH_LIB_PATH="):
            new_lines.append(fastpt_torch_lib_path_str + "\n")
            fastpt_torch_found = True
            continue
        if line.startswith("FASTPT_CUDA_LIB_PATH="):
            new_lines.append(fastpt_cuda_lib_path_str + "\n")
            fastpt_cuda_found = True
            continue
        if torch_path_env_str in line:
            torch_env_found = True
        if cuda_path_env_str in line:
            cuda_env_found = True
        new_lines.append(line)

    if not fastpt_torch_found and not fastpt_cuda_found and not torch_env_found and not cuda_env_found:
        new_lines.append(fastpt_torch_lib_path_str + "\n")
        new_lines.append(fastpt_cuda_lib_path_str + "\n")
        new_lines.append(torch_path_env_str + "\n")
        new_lines.append(cuda_path_env_str + "\n")


    if torch_env_found and cuda_env_found and (not fastpt_torch_found and not fastpt_cuda_found):
        for i, line in enumerate(new_lines):
            if torch_path_env_str in line or cuda_path_env_str in line:
                new_lines.insert(i, fastpt_torch_lib_path_str + "\n")
                new_lines.insert(i + 1, fastpt_cuda_lib_path_str + "\n")
                break


    try:
        with open(bashrc_path, 'w') as f:
            f.writelines(new_lines)
        if os.getenv("FASTPT_DEBUG", "0") == "1":
            if distro_name == "Kylin":
                print("Updated LD_LIBRARY_PATH and variables in /etc/profile")
            else:
                print("Updated LD_LIBRARY_PATH and variables in ~/.bashrc")
    except PermissionError:
        if distro_name == "Kylin":
            print("Permission denied, please add the following lines to /etc/profile manually:")
        else:
            print("Permission denied, please add the following lines to ~/.bashrc manually:")
        print(fastpt_torch_lib_path_str)
        print(fastpt_cuda_lib_path_str)
        print(torch_path_env_str)
        print(cuda_path_env_str)
    finally:
        if os.getenv("FASTPT_DEBUG", "0") == "1":
            if distro_name == "Kylin":
                print("Please run 'source /etc/profile' to make the changes effective")
            else:
                print("Please run 'source ~/.bashrc' to make the changes effective")



def cuda_exe():
    try:
        check_torch_version(required_version)
        print("torch version check success!")
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")

    required_so_files = ["libc10_cuda.so", "libtorch_cuda.so", "libtorch_mocker.so"]

    all_exist = all(os.path.exists(os.path.join(torch_lib_dir, so_file)) for so_file in required_so_files)
    torch_install_path_ = os.path.dirname(torch.__file__)
    if all_exist:
        print("torch mocker so 已存在")
    else:
        print("正在初始化 torch mocker so...")
        fastpt_install_path_ = os.path.dirname(__file__)

        target_dir = torch_install_path_
        source_dir = os.path.join(fastpt_install_path_, './torch')
        if os.path.exists(source_dir):
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    if any([file.endswith(file_end) for file_end in  ['.so']]):
                        src_file = os.path.join(root, file)
                        dst_file = os.path.join(target_dir, os.path.relpath(src_file, source_dir))
                        dst_dir  = os.path.dirname(dst_file)

                        # 创建目标目录（如果不存在）
                        os.makedirs(dst_dir, exist_ok=True)
                        # 文件备份
                        # if os.path.exists(dst_file) and not os.path.exists(dst_file + '.bak'):
                        #     print(f'Backup {dst_file} to {dst_file}.bak')
                        #     shutil.move(dst_file, dst_file + '.bak')
                        # 复制并覆盖文件
                        if os.getenv("FASTPT_DEBUG", "0") == "1":
                            print(f'{dst_file} is changed')

                        shutil.copy(src_file, dst_file)
        print("torch mocker so 已初始化")
    torch_lib_path = os.path.join(torch_install_path_, "lib")
    cuda_lib_path = os.getenv("ROCM_PATH", "/opt/dtk/")
    if dtk_version == '2504':
        cuda_lib_path_str  = os.path.join(cuda_lib_path, "cuda/lib64/")
    elif dtk_version == '25041':
        cuda_lib_path_str  = os.path.join(cuda_lib_path, "cuda/cuda-11/lib64/")
    else :
        cuda_lib_path_str  = os.path.join(cuda_lib_path, "cuda/cuda-12/lib64/")
    fastpt_torch_lib_path_str = f"FASTPT_TORCH_LIB_PATH={torch_lib_path}"
    fastpt_cuda_lib_path_str = f"FASTPT_CUDA_LIB_PATH={cuda_lib_path_str}"

    # 定义环境变量字符串
    torch_path_env_str = f"export LD_LIBRARY_PATH=$FASTPT_TORCH_LIB_PATH:$LD_LIBRARY_PATH"
    cuda_path_env_str = f"export LD_LIBRARY_PATH=$FASTPT_CUDA_LIB_PATH:$LD_LIBRARY_PATH"

    # 调试信息
    if os.getenv("FASTPT_DEBUG", "0") == "1":
        print(f"torch_lib_path    : {torch_lib_path}")
        print(f"cuda_lib_path     : {cuda_lib_path}")
        print(f"torch_path_env_str: {torch_path_env_str}")
        print(f"cuda_path_env_str : {cuda_path_env_str}")

    # 读取 .bashrc 文件内容
    bashrc_path = os.path.expanduser("~/.bashrc")  if distro_name != "Kylin" else os.path.abspath("/etc/profile")
    with open(bashrc_path, 'r') as f:
        lines = f.readlines()

    # 检查并删除所有重复的 FASTPT_TORCH_LIB_PATH 和 FASTPT_CUDA_LIB_PATH 定义
    new_lines = []
    fastpt_torch_found = False
    fastpt_cuda_found = False
    torch_env_found = False
    cuda_env_found = False

    for line in lines:
        if line.startswith("FASTPT_TORCH_LIB_PATH="):
            new_lines.append(fastpt_torch_lib_path_str + "\n")
            fastpt_torch_found = True
            continue
        if line.startswith("FASTPT_CUDA_LIB_PATH="):
            new_lines.append(fastpt_cuda_lib_path_str + "\n")
            fastpt_cuda_found = True
            continue
        if torch_path_env_str in line:
            torch_env_found = True
        if cuda_path_env_str in line:
            cuda_env_found = True
        new_lines.append(line)
    # 如果两个变量和两个环境变量设置都不存在，则直接写入
    if not fastpt_torch_found and not fastpt_cuda_found and not torch_env_found and not cuda_env_found:
        new_lines.append(fastpt_torch_lib_path_str + "\n")
        new_lines.append(fastpt_cuda_lib_path_str + "\n")
        new_lines.append(torch_path_env_str + "\n")
        new_lines.append(cuda_path_env_str + "\n")


    # 如果只存在 export 的两个设置，则在第一个 export 设置之前一行写入两个变量
    if torch_env_found and cuda_env_found and (not fastpt_torch_found and not fastpt_cuda_found):
        for i, line in enumerate(new_lines):
            if torch_path_env_str in line or cuda_path_env_str in line:
                new_lines.insert(i, fastpt_torch_lib_path_str + "\n")
                new_lines.insert(i + 1, fastpt_cuda_lib_path_str + "\n")
                break

    # 将修改后的内容写回 .bashrc 文件
    try:
        with open(bashrc_path, 'w') as f:
            f.writelines(new_lines)
        if os.getenv("FASTPT_DEBUG", "0") == "1":
            if distro_name == "Kylin":
                print("Updated LD_LIBRARY_PATH and variables in /etc/profile")
            else:
                print("Updated LD_LIBRARY_PATH and variables in ~/.bashrc")
    except PermissionError:
        if distro_name == "Kylin":
            print("Permission denied, please add the following lines to /etc/profile manually:")
        else:
            print("Permission denied, please add the following lines to ~/.bashrc manually:")
        print(fastpt_torch_lib_path_str)
        print(fastpt_cuda_lib_path_str)
        print(torch_path_env_str)
        print(cuda_path_env_str)
    finally:
        if os.getenv("FASTPT_DEBUG", "0") == "1":
            if distro_name == "Kylin":
                print("Please run 'source /etc/profile' to make the changes effective")
            else:
                print("Please run 'source ~/.bashrc' to make the changes effective")

      
# 卸载接口
def custom_uninstall():
    torch_install_path_ = os.path.dirname(torch.__file__)
    fastpt_install_path_ = os.path.dirname(__file__)

    target_dir = torch_install_path_
    source_dir = os.path.join(fastpt_install_path_, './torch')  # 源文件夹
    
    if os.path.exists(source_dir):
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if any([file.endswith(file_end) for file_end in include_ends + ['.so']]):
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(target_dir, os.path.relpath(src_file, source_dir))
                    dst_dir = os.path.dirname(dst_file)

                    backfile = dst_file + '.bak'
                    if os.path.exists(backfile):
                        print(f'Restore {backfile} to {dst_file}')
                        shutil.copy(backfile, dst_file)
                        os.remove(backfile)
                    else:
                        if os.path.exists(dst_file):
                            print(f'Remove {dst_file}')
                            os.remove(dst_file)

    else:
        print(f"source_dir {source_dir} not found")

