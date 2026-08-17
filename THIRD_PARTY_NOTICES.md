# Third-Party Notices

## Project provenance

FastPT consists of HYGON-developed core functionality and components derived
from and adapted from PyTorch.

- HYGON-developed portions include FastPT-specific architecture, integration,
  build, conversion, adaptation, and HCU/DTK support code, except where an
  upstream or third-party source is explicitly identified below.
- PyTorch-derived portions are based on PyTorch `v2.12.0`, fixed at Commit
  `0d62256a2b23365f8e1604297eb23a6545102aa8`, and have been adapted for
  FastPT, torch-mocker, DTK, and HCU compatibility.
- Upstream copyright and license notices are retained. Files containing
  substantive HYGON modifications carry the corresponding HYGON modification
  notice.
- Detailed local-to-upstream file mappings and classifications are recorded in
  `torch-mocker/PYTORCH_SOURCE_MAPPING.md`.

This notice distinguishes HYGON-developed code from PyTorch-derived and other
third-party components. It does not claim HYGON ownership of unmodified
upstream code.

## Third-party component registry

This file records the provenance and license information for upstream and
third-party components included in FastPT. Complete third-party license texts
are preserved in the license-text annex
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES), in component license files, and in
the component source files.

| Component | Source and fixed version | Copyright | License | Local path | HYGON changes |
| --- | --- | --- | --- | --- | --- |
| NVIDIA cuDNN Frontend | <https://github.com/NVIDIA/cudnn-frontend/tree/2533f5e5c1877fd76266133c1479ef1643ce3a8b>, version `1.6.1`, Commit `2533f5e5c1877fd76266133c1479ef1643ce3a8b` | Copyright (c) NVIDIA Corporation | MIT; bundled license at `torch-mocker/third_party/cudnn-frontend/LICENSE.txt` | `torch-mocker/third_party/cudnn-frontend/` | 64 of 66 originally bundled files are byte-identical to the fixed upstream Commit. `include/cudnn_frontend/graph_interface.h` and `include/cudnn_frontend_utils.h` each add an `nv_bfloat16` compatibility alias and carry HYGON Copyright and SPDX notices. Original NVIDIA notices are retained. |
| nlohmann/json | <https://github.com/nlohmann/json/tree/9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03>, version `3.11.3`, upstream Tag Commit `9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03`; bundled through the fixed cuDNN Frontend Commit above | Copyright (c) 2013-2023 Niels Lohmann | MIT; embedded Abseil utility is Apache-2.0 | `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/thirdparty/nlohmann/` | Embedded cuDNN Frontend copy; original notices retained. The bundled `json.hpp` is not claimed to be byte-identical to the standalone upstream release asset. |
| fmt | <https://github.com/fmtlib/fmt/tree/407c905e45ad75fc29bf0f9bb7c5c2fd3475976f>, Commit `407c905e45ad75fc29bf0f9bb7c5c2fd3475976f` | Copyright (c) 2012-present Victor Zverovich and fmt contributors | MIT | `torch-mocker/third_party/fmt` | Unmodified pinned Git submodule |
| PyTorch TunableOp-derived code | Fixed PyTorch v2.12.0 sources: <https://github.com/pytorch/pytorch/blob/0d62256a2b23365f8e1604297eb23a6545102aa8/aten/src/ATen/cuda/tunable/StreamTimer.cpp> and <https://github.com/pytorch/pytorch/blob/0d62256a2b23365f8e1604297eb23a6545102aa8/aten/src/ATen/cuda/tunable/GemmCommon.h>; Commit `0d62256a2b23365f8e1604297eb23a6545102aa8` | Copyright (c) Microsoft Corporation and Advanced Micro Devices, Inc. | MIT, as declared in the retained upstream source headers | `torch-mocker/aten/src/ATen/cuda/tunable/StreamTimer.cpp`, `torch-mocker/include/ATen/cuda/tunable/GemmCommon.h` | Derived from the fixed PyTorch TunableOp implementation and adapted for torch-mocker/FastPT. The transitive ONNX Runtime attribution present in the upstream headers, and the Microsoft and AMD notices, are retained. |
| PyTorch-derived runtime and headers | <https://github.com/pytorch/pytorch/tree/0d62256a2b23365f8e1604297eb23a6545102aa8>, version `2.12.0`, Commit `0d62256a2b23365f8e1604297eb23a6545102aa8` | Copyright (c) PyTorch contributors and other respective copyright holders | BSD-3-Clause; bundled license at `torch-mocker/LICENSE.pytorch.txt` | PyTorch-derived portions of `torch-mocker/aten/`, `torch-mocker/c10/`, `torch-mocker/include/`, and `torch-mocker/torch/` | Adapted for FastPT, DTK, and HCU compatibility. Upstream files without per-file notices remain attributed through the bundled PyTorch license and this notice; existing upstream notices are retained. |
| PyTorch HIPIFY | <https://github.com/pytorch/pytorch/blob/0d62256a2b23365f8e1604297eb23a6545102aa8/torch/utils/hipify/hipify_python.py>, version `2.12.0`, Commit `0d62256a2b23365f8e1604297eb23a6545102aa8`; imported FastPT repository snapshot `5e0686df773fe13efb608c5264097464e31e7564` | Copyright (c) Advanced Micro Devices, Inc. and Facebook, Inc. | MIT | `fastpt/hipify_python.py` | Adapted for FastPT mappings; AMD and Facebook notices plus the HYGON modification notice are retained |


### NVIDIA cuDNN Frontend file registry

The following per-file paths are listed for deterministic compliance-tool recognition. They are covered by the fixed NVIDIA cuDNN Frontend component entry above.

<details>
<summary>NVIDIA cuDNN Frontend file registry（66 files）</summary>

- `torch-mocker/third_party/cudnn-frontend/include/cudnn_backend_base.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/backend/backend_descriptor.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/backend/execution_helpers.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/backend/plan_helpers.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/context.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/cudnn_interface.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/graph_helpers.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/graph_interface.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/graph_properties.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/batchnorm.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/batchnorm_inference.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/bn_finalize.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/conv_dgrad.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/conv_fprop.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/conv_wgrad.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/dbn.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/dbn_weight.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/dln.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/genstats.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/instancenorm.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/layernorm.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/matmul.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/matmul_fp8.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/pointwise.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/reduction.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/resample.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/reshape.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/rmsnorm.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/rng.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/scaled_dot_product_flash_attention.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/sdpa_fp8.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/sdpa_fp8_bwd.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/slice.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node/softmax.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/node_interface.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/plans.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/thirdparty/nlohmann/LICENSE.MIT`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/thirdparty/nlohmann/json.hpp`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend/utils/serialize.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_ConvDesc.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_Engine.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_EngineConfig.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_EngineConfigGenerator.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_EngineFallbackList.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_Errata.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_ExecutionPlan.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_ExecutionPlanCache.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_Filters.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_Heuristics.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_Logging.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_MatMulDesc.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_Operation.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_OperationGraph.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_PointWiseDesc.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_ReductionDesc.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_Reorder_Tensor.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_Resample.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_Rng.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_Tensor.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_VariantPack.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_find_plan.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_get_plan.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_shim.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_utils.h`
- `torch-mocker/third_party/cudnn-frontend/include/cudnn_frontend_version.h`

</details>


### PyTorch-derived file classification

The PyTorch-derived and generated files bundled in `torch-mocker` are mapped
individually in `torch-mocker/PYTORCH_SOURCE_MAPPING.md`. Comparison against
the fixed PyTorch baseline above produced the following classifications:

- 17 files are byte-identical to the fixed upstream baseline and retain
  PyTorch attribution without a HYGON copyright notice (H5).
- 88 files contain substantive FastPT, torch-mocker, DTK, HCU, HIP/ROCm,
  API-adaptation, or source-reduction modifications and carry the HYGON
  modification notice while retaining PyTorch attribution and the
  BSD-3-Clause license (H2). Of these, 49 contain direct FastPT/HCU/DTK/mocker
  markers; modification ownership for the other 39 is confirmed by internal
  torch-mocker/FastPT introduction and adaptation history.
- 3 files are generated artifacts and are not treated as HYGON-authored source
  files (H6).

## Generated artifacts

The following generated files are shipped from the fixed FastPT repository
snapshot `5e0686df773fe13efb608c5264097464e31e7564`:

Three of these files (`CUDAConfig.h`, `Caffe2Targets-release.cmake`, and
`torch/version.py`) are included in the 108-file PyTorch source mapping above.
The other two generated artifacts are outside that mapping scope.

- `torch-mocker/include/c10/cuda/impl/cuda_cmake_macros.h` is an automatically
  generated C10 build header. The repository does not contain its generator.
- `torch-mocker/torch/share/cmake/Caffe2/Caffe2Targets.cmake` is a CMake export
  file copied into the package by `setup.py`, `fastpt/fastpt`, and
  `fastpt/init_process.py`. The repository does not contain the originating
  CMake export build tree.
- `torch-mocker/torch/share/cmake/Caffe2/Caffe2Targets-release.cmake` is the
  corresponding generated Release-configuration import file.
- `torch-mocker/torch/version.py` is generated PyTorch package version metadata.
- `torch-mocker/include/ATen/cuda/CUDAConfig.h` is generated PyTorch CUDA build
  configuration metadata.

These generated artifacts are not treated as HYGON-authored source files and
must not receive mechanically added source headers. Reproduction requires the
corresponding fixed build environment or generator source.

The PyTorch-derived runtime, headers, and HIPIFY source are registered against
the fixed PyTorch `v2.12.0` Commit above. The exact local FastPT snapshot and
HYGON adaptation scope are recorded separately so the public source baseline
and local modifications are not conflated.
