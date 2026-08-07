// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/native/cuda/thread_constants.h>

#include "../../support/TestHarness.h"

namespace {

template <typename Function>
at::Tensor make_iterator_output(const at::Tensor& input, Function&& launch) {
  auto output = at::empty_like(input);
  auto iterator = at::TensorIteratorConfig().add_output(output).add_input(input).build();
  launch(iterator);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  return output;
}

} // namespace

TM_TEST(ATenCUDA, CUDALoops) {
  const auto input = at::randn({256}, at::device(at::kCUDA).dtype(at::kFloat));

  const auto squared = make_iterator_output(input, [](at::TensorIterator& iterator) {
    auto function = [] GPU_LAMBDA(float value) { return value * value; };
    using function_t = decltype(function);
    constexpr int tensor_count = function_traits<function_t>::arity + 1;
    std::array<char*, tensor_count> data;
    for (int index = 0; index < tensor_count; ++index) {
      data[index] = static_cast<char*>(iterator.data_ptr(index));
    }
    at::native::launch_vectorized_kernel<function_t>(iterator.numel(), function, data);
  });
  TM_REQUIRE(at::allclose(squared.cpu(), (input * input).cpu()));

  const auto doubled = make_iterator_output(input, [](at::TensorIterator& iterator) {
    auto function = [] GPU_LAMBDA(float value) { return value * 2.0f; };
    using function_t = decltype(function);
    using traits = function_traits<function_t>;
    constexpr int tensor_count = traits::arity + 1;
    std::array<char*, tensor_count> data;
    for (int index = 0; index < tensor_count; ++index) {
      data[index] = static_cast<char*>(iterator.data_ptr(index));
    }
    at::native::launch_unrolled_kernel<function_t>(
        iterator.numel(),
        function,
        data,
        TrivialOffsetCalculator<traits::arity>(),
        TrivialOffsetCalculator<1>(),
        at::native::memory::LoadWithCast<traits::arity>(iterator),
        at::native::memory::StoreWithCast<1>(iterator));
  });
  TM_REQUIRE(at::allclose(doubled.cpu(), (input * 2.0f).cpu()));

  const auto legacy_doubled = make_iterator_output(input, [](at::TensorIterator& iterator) {
    auto function = [] GPU_LAMBDA(float value) { return value * 2.0f; };
    using function_t = decltype(function);
    using traits = function_traits<function_t>;
    using value_t = typename traits::result_type;
    constexpr int tensor_count = traits::arity + 1;
    std::array<char*, tensor_count> data;
    for (int index = 0; index < tensor_count; ++index) {
      data[index] = static_cast<char*>(iterator.data_ptr(index));
    }
    const auto offset_calculator = ::make_offset_calculator<tensor_count>(iterator);
    constexpr int unroll_factor = sizeof(value_t) >= 4 ? 2 : 4;
    at::native::launch_legacy_kernel<128, unroll_factor>(
        iterator.numel(),
        [=] GPU_LAMBDA(int index) {
          const auto offsets = offset_calculator.get(index);
          auto* output = reinterpret_cast<value_t*>(data[0] + offsets[0]);
          *output = at::native::invoke(function, &data[1], &offsets[1], 1);
        });
  });
  TM_REQUIRE(at::allclose(legacy_doubled.cpu(), (input * 2.0f).cpu()));
}
