# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.build import build
from setuptools.command.install import install
from wheel.bdist_wheel import bdist_wheel
import os
import subprocess
import sys
import shutil
try:
    import torch

    torch_install_path = os.path.dirname(torch.__file__)
    # CMake consumes TORCH_PATH from its environment. Derive it from the
    # imported torch package so callers do not need to export it manually.
    os.environ["TORCH_PATH"] = torch_install_path
except ImportError as e:
    print("Torch is required, ", e)
    raise RuntimeError("Torch is required") from e

include_ends=['.h', '.cuh', '.hpp']

install_flag = True


class CustomInstall(install):
    def run(self):
        # 首先运行默认的安装逻辑
        install.run(self)
        if not getattr(self, 'skip_install_flag', False):
            print("Skiped install")
            return
            # 覆盖同名文件
        import torch

        torch_install_path_ = os.path.dirname(torch.__file__)
        target_dir = torch_install_path_ #
        # target_dir = os.path.join(self.install_lib, 'torch')
        source_dir = 'torch'  # 源文件夹
        
        if os.path.exists(source_dir):
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    if any([file.endswith(file_end) for file_end in include_ends+['.so']]):
                        src_file = os.path.join(root, file)
                        dst_file = os.path.join(target_dir, os.path.relpath(src_file, source_dir))
                        dst_dir = os.path.dirname(dst_file)
                        
                        # 创建目标目录（如果不存在）
                        os.makedirs(dst_dir, exist_ok=True)
                        
                        # 复制并覆盖文件
                        print(f'Copying {src_file} to {dst_file}')
                        shutil.copy(src_file, dst_file)


class CMakeBuild(build_ext):
    def run(self):
        # 创建构建目录
        if not os.path.exists('build'):
            os.makedirs('build')
        
        # [新增] 在构建阶段创建Python文件目标目录
        py_dest_dir = os.path.join('build', 'lib', 'torch')
        os.makedirs(py_dest_dir, exist_ok=True)
        os.makedirs(os.path.join(py_dest_dir, 'cuda'), exist_ok=True)
        os.makedirs(os.path.join(py_dest_dir, 'utils'), exist_ok=True)

        # [新增] 拷贝指定的Python文件到构建目录
        py_files = [
            ('./torch/version.py', 'version.py'),
            ('torch/cuda/__init__.py', 'cuda/__init__.py'),
            ('torch/utils/cpp_extension.py', 'utils/cpp_extension.py')
        ]
        for src, dest in py_files:
            dest_path = os.path.join(py_dest_dir, dest)
            shutil.copy(src, dest_path)
            print(f"Copying {src} to build directory: {dest_path}")

        # CMake inherits TORCH_PATH from the torch package imported above.
        torch_lib_path = torch_install_path
        if not os.path.isdir(torch_lib_path):
            raise RuntimeError(f"Torch installation directory does not exist: {torch_lib_path}")
        cmake_args = os.environ.get("TORCH_MOCKER_CMAKE_ARGS", "")
        make_args = "-j8 VERBOSE=1"
        make_args_env = os.environ.get("TORCH_MOCKER_MAKE_ARGS", "")
        make_args += make_args_env
        extra_cmake_args = ""
        cmake_args+=extra_cmake_args
        build_cmd = f"""
            cd build
            cmake {cmake_args} ..
            make {make_args}
        """
        ret = os.system(build_cmd)
        if ret != 0:
            raise Exception("Failed to build torch-mocker")


        # 创建目标目录
        torch_lib_dir = os.path.join('torch', 'lib')
        torch_include_dir = os.path.join('torch','include')
        os.makedirs(torch_lib_dir, exist_ok=True)
        os.makedirs(torch_include_dir, exist_ok=True)

        # 复制生成的 .so 文件到 torch/lib
        so_files = ["libc10_cuda.so", "libtorch_cuda.so", "libtorch_mocker.so"]
        for so_file_name  in so_files:
            so_file = os.path.join('build', so_file_name)  # 确保路径正确
            shutil.copy(so_file, torch_lib_dir)

        # 复制 include 文件到 torch/include
        file_paths = []
        include_dir = 'include'
        for root, dirs, files in os.walk(include_dir):
            for filename in files:
                # 拼接完整路径
                full_path = os.path.join(root, filename)
                file_paths.append(full_path)
        for header in file_paths:
            if any([header.endswith(header_end) for header_end in include_ends]):
                target_directory = os.path.dirname(os.path.join("torch", header))
                os.makedirs(target_directory, exist_ok=True)
                shutil.copy(header, os.path.join("torch", header))
        print("\n torch mocker build successed\n")

         
class CustomInstall(install):
    def run(self):
        # 首先运行默认的安装逻辑
        install.run(self)
        
        if not getattr(self, 'skip_install_flag', False):
            print("Skip install")
            return

        # [新增] 安装Python文件
        import torch
        torch_install_path_ = os.path.dirname(torch.__file__)
        
        # 需要安装的三个Python文件路径
        py_files = [
            ('build/lib/torch/version.py', os.path.join(torch_install_path_, 'version.py')),
            ('build/lib/torch/cuda/__init__.py', os.path.join(torch_install_path_, 'cuda', '__init__.py')),
            ('build/lib/torch/utils/cpp_extension.py', os.path.join(torch_install_path_, 'utils', 'cpp_extension.py'))
        ]
        
        for src, dst in py_files:
            if os.path.exists(src):
                dst_dir = os.path.dirname(dst)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy(src, dst)
                print(f'Successfully installed {src} -> {dst}')
        target_dir = torch_install_path_ #
        # target_dir = os.path.join(self.install_lib, 'torch')
        source_dir = 'torch'  # 源文件夹
        
        if os.path.exists(source_dir):
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    if any([file.endswith(file_end) for file_end in include_ends+['.so']]):
                        src_file = os.path.join(root, file)
                        dst_file = os.path.join(target_dir, os.path.relpath(src_file, source_dir))
                        dst_dir = os.path.dirname(dst_file)
                        
                        # 创建目标目录（如果不存在）
                        os.makedirs(dst_dir, exist_ok=True)
                        
                        # 复制并覆盖文件
                        print(f'Copying {src_file} to {dst_file}')
                        shutil.copy(src_file, dst_file)


class CustomBuild(build):
    def run(self):
        # 先调用 build_ext
        CMakeBuild.run(self)
        # CMakeBuild.run(self)
        # 再继续常规的 build 流程
        build.run(self)

class CustomDistWheel(bdist_wheel): 
    def run(self): 
        self.run_command('build') 
        self.skip_install = True
        bdist_wheel.run(self)

class NuitkaBuild(build_ext):
    def run(self):
        # 定义要编译的文件,fastpt 将同名的二级目录拷贝到这里
        target_file = os.listdir("fastpt")
        target_dir = 'fastpt'
        target_dir = 'build/lib/'
        # 使用 Nuitka 进行编译
        compile_cmd = f"""
            nuitka --module 'fastpt'  --include-package='fastpt' --output-dir={target_dir} 
        """
        ret = os.system(compile_cmd)
        shutil.rmtree('build/lib/fastpt.build')


class Lib_build(NuitkaBuild, CMakeBuild):
    def run(self):
        # NuitkaBuild.run(self)
        CMakeBuild.run(self)

package_data = {}
#package_data["torch"] = ["include/**/*", "lib/*"]
# package_data['fastpt']= ['build/fastpt/*.so', 'build/fastpt/*.pyi']
package_data["torch"] = [
    "include/**/*",
    "lib/*",
    "version.py",          # 新增Python文件
    "cuda/__init__.py",    # 新增Python文件
    "utils/cpp_extension.py"  # 新增Python文件
]

setup(
    name='fastpt',
    version='0.1',
    packages=['torch',],  # 指定包名称为 torch
    package_dir={'torch': 'torch'},#, 'fastpt':'fastpt'},  # 指定包目录
    package_data=package_data,
    ext_modules=[], # 需要的源文件
    include_package_data=True,
    cmdclass={'build': CustomBuild,
              'bdist_wheel':CustomDistWheel,
              'install': CustomInstall,
              },
    zip_safe=False,
)
