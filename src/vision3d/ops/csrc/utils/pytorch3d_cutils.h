/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torch/headeronly/util/Exception.h>

#define CHECK_CUDA(x) \
  STD_TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor.")
#define CHECK_CONTIGUOUS(x) \
  STD_TORCH_CHECK((x).is_contiguous(), #x " must be contiguous.")
#define CHECK_CONTIGUOUS_CUDA(x) \
  CHECK_CUDA(x);                 \
  CHECK_CONTIGUOUS(x)
#define CHECK_CPU(x) \
  STD_TORCH_CHECK(   \
      (x).is_cpu(), "Cannot use CPU implementation: " #x " not on CPU.")
