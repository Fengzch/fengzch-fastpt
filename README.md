## 介绍

`FastPT` 是一款基于 `Python` 的应用编译工具。借助 `FastPT`，开发人员可以在 `HCU` 上开发与部署基于 `PyTorch`、且包含 `CUDA` 代码的应用。它支持以下两种典型方式：

- 直接编译 `CUDA` 源码，无需转码。
- 先将源码转换为 `HIP` 格式，再通过 `hipcc` 完成适配编译。

## 编译与安装

`FastPT` 的 wheel 会构建集成的 `torch-mocker`，因此构建环境必须同时具备 DTK、已安装的目标 HIP torch package 和同版本 HIP torch 源码。`TORCH_PATH` 必须指向干净的 HIP torch 安装目录，不能是已经被 FastPT `-C` 覆盖过的 torch。

构建前确认：

- FastPT 分支、已安装 torch 和 `HIP_TORCH_PATH` 对应同一 PyTorch 版本；
- 已安装 `wheel`、CMake、GNU Make 和可用的 DTK 编译环境；
- 当前环境未安装过 `FastPT wheel`。检查torch的安装路径下， torch/lib 不存在 libtorch_cuda.so、libc10_cuda.so、libtorch_mocker.so，并确认`torch/utils/cpp_extension.py` 不含 USE_FASTPT_CUDA 等 FastPT 路由。

### 构建 wheel

```bash
cd /path/to/fastpt
source /opt/dtk/cuda/env.sh

# 必须是干净的目标 HIP torch package，
export TORCH_PATH="$(python3 -c 'import os, torch; print(os.path.dirname(torch.__file__))')"
# /path/to/pytorch: 为HCU适配的pytorch的源码，注意版本分支。
export HIP_TORCH_PATH=/path/to/pytorch

git submodule update --init --recursive
rm -rf build dist fastpt.egg-info torch-mocker/build

# FASTPT_CTEST可以控制CTest部分的编译，默认为 1：构建 torch-mocker 和已注册的 CTest 可执行文件，不需要Ctest可以设置为0.
python3 setup.py bdist_wheel
```

构建产物位于 `dist/fastpt-*.whl`。若仅需要构建发布 wheel，且已通过后文的独立 CTest，可设置 `FASTPT_CTEST=0` 跳过测试可执行文件构建。

### 仅打包转码 CLI

只需要 `fastpt.hipify`、`fastpt.hipify.file`、`fastpt.hipify.folder` 和 `fastpt.hipify.cmake` 时，可跳过 torch-mocker 构建：

```bash
cd /path/to/fastpt
FASTPT_HIPIFY_ONLY=1 python3 setup.py bdist_wheel
```

该模式不需要设置 `TORCH_PATH` 或 `HIP_TORCH_PATH`，但仍需要可导入的匹配 HCU/HIP torch，以生成包版本元数据。它不构建或打包 `libc10_cuda.so`、`libtorch_cuda.so`、`libtorch_mocker.so`、torch overlay 和 `fastpt -C/-E` 环境脚本，只能用于源码/CMake 转码, `CUDAExtension`、`CppExtension`的扩展编译，不能用于基于GPUFusion的不转码 编译或运行环境初始化。

### 安装与验证

```bash
python3 -m pip install --no-deps --force-reinstall dist/fastpt-*.whl
python3 -c "import fastpt; print(fastpt.__version__)"

fastpt.hipify.file --help
```

上述检查只验证 wheel 安装、Python import 和 hipify CLI 入口。它不会验证 CUDA-facing 编译链路。

### CTest 测试

`FASTPT_CTEST`不设置时默认为1，会编译CTest测试文件，可通过以下方式进行`torch-mocker`部分的测试：

```bash
export LD_LIBRARY_PATH="$PWD/torch-mocker/build:$TORCH_PATH/lib:${LD_LIBRARY_PATH:-}"
ctest --test-dir torch-mocker/build --output-on-failure -j1
```

使用 `ctest --test-dir torch-mocker/build_ctest -N` 可列出已注册测试。定位单个失败时，使用测试名筛选，例如：

```bash
ctest --test-dir torch-mocker/build_ctest --output-on-failure \
  -R '^torch_mocker\.c10\.cuda_functions$'
```

## 使用说明

`FastPT` 安装完成后，可通过 `source /path/to/fastpt -X` 进行环境设置。需要注意：

- 通过 `which fastpt` 确认其`fastpt`的实际安装路径；
- `X` 为模式参数，不同参数对应不同使用场景。

| 使用场景   | 模式参数`X`    | 示例                                | 说明                                                                                                                                                                                                                                                  |
| ---------- | ---------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 不转码编译 | `-C` 或 `-c` | `source /usr/local/bin/fastpt -C` | 用于不转码编译场景下的环境初始化。 由于编译模式需要设置部分环境变量，因此每次在新的终端中进行编译前，都需要先执行该命令。                                                                                                                             |
|            | `-E` 或 `-e` | `source /usr/local/bin/fastpt -E` | 用于不转码编译产物的运行环境初始化。 当工程迁移到新环境后，安装 FastPT 并执行一次该命令即可。通常无需重复执行，只需保证当前系统中已完成初始化。                                                                                                       |
| 转码编译   | `-T` 或 `-t` | `source /usr/local/bin/fastpt -T` | 用于将CUDA代码转换成HIP代码，使用CUDAExtension、CppExtension做编译时，替换成fastpt中的接口。 该模式仅用于组件或程序的编译阶段，运行阶段通常不需要额外环境配置。适用于CUDAExtension/CPPExtension编译扩展实现的。只进行转码操作时，不需要这个指令设置。 |
| 帮助       | `-H` 或 `-h` | `source /usr/local/bin/fastpt -H` | 查看命令说明与使用方法。                                                                                                                                                                                                                              |

### 不转码编译

#### 使用方法

- 编译模式：执行 `source /usr/local/bin/fastpt -C` 初始化 `FastPT` 编译环境后，进入不转码编译模式，直接使用 `CUDA` 源码进行构建。编译处理按照组件或应用官方提供的编译流程执行。
- 执行模式：编译后的组件以 `whl` 包形式部署到安装有 `FastPT` 的新环境中时，需要确保已执行过运行环境初始化；否则可能出现找不到 `CUDA` 相关动态库的错误。执行 `source /usr/local/bin/fastpt -E` 即可完成该初始化。

#### 不转码编译注意事项

- `FastPT` 适用于依赖 `torch` 生态的组件，其版本必须与 `torch` 版本严格对应；
- 不支持依赖 `cutlass` 或包含内嵌汇编指令的组件编译。如果组件中存在内嵌汇编指令，建议改用 `C/C++` 接口进行替换；
- 在编译模式 `(-C)` 与执行模式 `(-E)` 下，`torch.version.cuda` 与 `torch.version.hip` 会被分别设置。少部分应用在运行时会依赖这两个变量，因此需要根据实际情况在应用侧进行适配，并区分编译环境与运行环境的设置；
- 不要使用`nv`编译的包，当触发 `CUDAHooksMocker` 重复注册问题时，请检查环境下是否有安装`CU`编译支持的包；
- `C++` 标准要求为 `C++17` 或更高版本；
- 当前不支持 `cudart` 静态库链接。如果存在 `-lcudart_static` 或 `CMAKE_CUDA_RUNTIME_LIBRARY=Static`，可改为 `-lcudart`，或将 `CMAKE_CUDA_RUNTIME_LIBRARY` 设置为 `Shared`。对于其他不支持的编译指令，可视情况屏蔽；
- 若出现 `CUDA` 与 `HIP` 符号定义冲突，通常说明代码同时编译了 `CUDA` 与 `HIP` 源文件。常见原因如下：
  - 编译前未执行 `-C`，导致代码曾按转码编译方式处理并生成了 `HIP` 代码。
  - 已执行 `-C`，但之后又执行了 `-E`，导致不转码编译所需的部分环境变量缺失，进而误走转码编译流程。

### CUDAExtension/CppExtension 编译

转码编译模式通过对代码及编译指令进行转换，实现 `CUDA -> HIP` 的映射，并在 `HIP` 环境下完成编译。该模式下，工具对 `PyTorch` 中的 `CUDAExtension`、`CppExtension`、`hipify` 等接口提供了面向 `HIP` 环境的补充与优化。
 在使用'pytorch'中的 `CUDAExtension/CppExtension` 进行编译构建时，如果出现指令不支持的情况，可以尝试使用此`FastPT`中的接口。

#### 使用方法

转码编译适用于通过 `setup.py` 使用 `CUDAExtension`、`CppExtension` 构建组件的场景。使用时，执行 `source /usr/local/bin/fastpt -T` 启用`FastPT`下的两个扩展接口，编译使用之前的编译处理方式。
 通过 `CUDAExtension/CppExtension` 做编译实现的，如果使用torch中的接口不能进行覆盖的，可以尝试使用 `FastPT`

### 代码转换

代码转码通过 `fastpt.hipify.*` 命令完成，不需要执行 `source /usr/local/bin/fastpt -T`。`-T` 仅用于 `setup.py` 中的 `CUDAExtension` / `CppExtension` 自动转码构建。

| 命令                     | 用途                                       |
| ------------------------ | ------------------------------------------ |
| `fastpt.hipify.file`   | 转换单个源文件                             |
| `fastpt.hipify.folder` | 转换完整源码树，并做基础 CMake 替换        |
| `fastpt.hipify.cmake`  | 检查或原地补齐已有 HIP 源码树的 CMake 配置 |

| 参数       | `fastpt.hipify.file`      | `fastpt.hipify.folder`    | `fastpt.hipify.cmake` | 说明                                                                                 |
| ---------- | --------------------------- | --------------------------- | ----------------------- | ------------------------------------------------------------------------------------ |
| 输入路径   | `source_file`             | `project_directory`       | `--repo`              | `file` 和 `folder` 的路径位置参数必填；`--repo` 默认当前目录。                 |
| 输出路径   | `-o`, `--output`        | `-o`, `--output`        | 不支持                  | `folder` 未指定输出时使用 `<project_directory>_dtk`。                            |
| 原地处理   | `--inplace`               | `--inplace`               | `--apply`             | `file` / `folder` 的原地模式会生成 `.dtk_bak`；CMake 仅在 `--apply` 时写入。 |
| 自定义映射 | `--custom`                | `--custom`                | 不支持                  | 映射文件默认名为`custom_hipify_mappings.json`。                                    |
| 忽略路径   | 不支持                      | `--ignore`                | 不支持                  | 支持单路径、逗号/分号/空白分隔路径，或每行一个路径的列表文件。                       |
| 详细输出   | `-v`, `--show_detailed` | `-v`, `--show_detailed` | 不支持                  | 输出 hipify 统计信息。                                                               |
| HIP 架构   | 不支持                      | 不支持                      | `--arch`              | 不指定时，默认读取`FASTPT_DTK_ARCH`环境变量，默认值为 `gfx928;gfx936;gfx938`。   |

`CUDA_HOME` 会自动改为 `ROCM_HOME`，`CUDA_PATH` 会自动改为 `ROCM_PATH`，例如 `${CUDA_HOME}/include` 变为 `${ROCM_HOME}/include`。硬编码路径（如 `/usr/local/cuda/include`）不会自动替换，CLI 会输出风险提示，需根据项目实际 DTK 路径手工处理。

#### 完整源码树示例

以下示例将 CUDA 工程转换到新目录，并补齐 CMake 的 HIP 架构和已识别依赖包配置：

```bash
fastpt.hipify.folder /workspace/app -o /workspace/app_dtk

# 先预览 CMake 差异，再原地应用
fastpt.hipify.cmake --repo /workspace/app_dtk --arch gfx936
fastpt.hipify.cmake --repo /workspace/app_dtk --arch gfx936 --apply

cmake -S /workspace/app_dtk -B /workspace/app_dtk/build \
  -DCMAKE_HIP_ARCHITECTURES=gfx936
cmake --build /workspace/app_dtk/build -j"$(nproc)"
```

`folder` 保持目录结构，并将参与转码的 `.cu` 文件生成对应的 `.hip` 文件。它会进行 CMake 的语法、源文件名和链接 target 替换；顶层 `find_package(...)`、`CMAKE_HIP_ARCHITECTURES` 配置以及 `.hip` 源路径告警由后续的 `hipify.cmake` 处理。

注意：

* CMake的转义处理可能不会覆盖所有接口，可能需要开发者自己补充些实现；
* third_party部分的代码默认不会被处理，需要转码时请单独进行处理；
* `--arch` 是可选参数，默认支持`gfx928,gfx936,gfx938`,可以根据实际情况进行设置。

#### 单文件示例

```bash
# 写入新文件，保留原 kernel.cu
fastpt.hipify.file kernel.cu -o kernel.hip

# 生成 kernel.hip；将 kernel.cu 备份为 kernel.cu.dtk_bak 
fastpt.hipify.file kernel.cu --inplace
```

`-o` 与 `--inplace` 同时指定时，`--inplace` 优先，`-o` 会被忽略。注意使用 `-o kernel.cu` 会覆盖原文件，因为该方式不生成备份。

#### 仅处理 CMake

`fastpt.hipify.cmake` 不转源码、不复制目录。默认只输出 diff 和风险提示；只有 `--apply` 才会修改 CMake 文件，并对修改文件创建 `.dtk_bak` 备份。

```bash
# 已有 HIP 源码树，只检查 CMake
fastpt.hipify.cmake --repo /workspace/app_dtk --arch gfx936

# 确认后原地写入
fastpt.hipify.cmake --repo /workspace/app_dtk --arch gfx936 --apply
```

如果 `.cu -> .hip` 的路径无法静态确定，或对应 `.hip` 文件不存在，命令会给出警告但不会拒绝 `--apply`。应以实际 CMake configure 和编译结果为准。

#### 自定义映射

内置规则未覆盖的 API 或标识符可通过 JSON 映射补充。例如在工程目录创建 `custom_hipify_mappings.json`：

```json
{
  "custom_map": {
    "cudaProjectHelper": "hipProjectHelper"
  }
}
```

调用时显式指定该文件：

```bash
fastpt.hipify.folder /workspace/app -o /workspace/app_dtk \
  --custom /workspace/app/custom_hipify_mappings.json
```

#### 转码编译注意事项

- `FastPT` 适用于依赖 `torch` 生态的组件，其版本必须与 `torch` 版本严格对应。
- 在执行 `source /path/to/fastpt -T` 后，`CUDAExtension`、`CppExtension`、`hipify` 会自动替换为 `FastPT` 对应接口。
- 转码编译模式下，无需执行 `source /opt/dtk/cuda/env.sh` 来启用 `gpufusion` 环境。
- 某些接口在 `CUDA` 环境中没有实现，组件侧可能自行补充；而 `HIP` 环境中可能已存在对应实现，这时会出现重复定义问题，可通过 `__HIP_PLATFORM_HCC__` 进行条件编译控制。
- `fastpt.hipify.cmake` 可处理常见 CMake 语法、`.cu/.hip` 文件名、CUDA 链接 target 和部分依赖包；复杂 CMake 宏、CUTLASS、动态变量与 `make` 规则仍需按实际编译错误适配。
- 如果工程中包含第三方依赖库，第三方库可能不会被自动处理，此时需要单独适配。
- 如果不希望引入 `ATen/dtk_macros.h` 头文件，可通过以下方式屏蔽：`export FASTPT_DTK_MACROS=1`。

## License

FastPT is distributed under the BSD 3-Clause License. See [LICENSE.txt](LICENSE.txt).

Third-party components and their license notices are listed in
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
