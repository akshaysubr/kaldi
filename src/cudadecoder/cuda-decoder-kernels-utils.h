// cudadecoder/cuda-decoder-kernels-utils.h
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Hugo Braun, Justin Luitjens, Ryan Leary
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_DECODER_CUDA_DECODER_KERNELS_UTILS_H_
#define KALDI_DECODER_CUDA_DECODER_KERNELS_UTILS_H_

#include "util/stl-utils.h"

namespace kaldi {
namespace CudaDecode {

// 1:1 Conversion float <---> sortable int
// We convert floats to sortable ints in order
// to use native atomics operation, which are
// way faster than looping over atomicCAS
__device__ __forceinline__ int32 binsearch_maxle(const int32 *vec,
                                                 const int32 val, int32 low,
                                                 int32 high) {
  while (true) {
    if (low == high) return low;  // we know it exists
    if ((low + 1) == high) return (vec[high] <= val) ? high : low;

    int32 mid = low + (high - low) / 2;

    if (vec[mid] > val)
      high = mid - 1;
    else
      low = mid;
  }
}

__device__ __forceinline__ int32 floatToOrderedInt(float floatVal) {
  int32 intVal = __float_as_int(floatVal);
  return (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF;
}

__device__ __forceinline__ float orderedIntToFloat(int32 intVal) {
  return __int_as_float((intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF);
}

struct MinPlus {
  __device__ int2 operator()(const int2 &a, const int2 &b) const {
    int2 c;
    c.x = min(a.x, b.x);
    c.y = a.y + b.y;
    return c;
  }
};

struct PlusPlus {
  __device__ int2 operator()(const int2 &a, const int2 &b) const {
    int2 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
  }
};

union UInt64UnionInt2 {
  int2 i2;
  unsigned long long int ull;
};

__device__ __inline__ int2 atomicAddI2(int2 *ptr, int2 val) {
  unsigned long long int *ptr64 =
      reinterpret_cast<unsigned long long int *>(ptr);
  UInt64UnionInt2 uval, uold;
  uval.i2 = val;
  uold.ull = atomicAdd(ptr64, uval.ull);
  return uold.i2;
}

// TODO use native atomicMin64
__device__ __inline__ void atomicMinI2(int2 *ptr, int2 val) {
  unsigned long long int *ptr64 =
      reinterpret_cast<unsigned long long int *>(ptr);
  UInt64UnionInt2 old, assumed, value;
  old.ull = *ptr64;
  value.i2 = val;
  if (old.i2.x <= val.x) return;
  do {
    assumed = old;
    old.ull = atomicCAS(ptr64, assumed.ull, value.ull);
  } while (old.ull != assumed.ull && old.i2.x > value.i2.x);
}

__device__ void atomicSub(int2 *ptr, int2 sub) {
  unsigned long long int *ptr64 =
      reinterpret_cast<unsigned long long int *>(ptr);
  UInt64UnionInt2 old, assumed, value;
  old.ull = *ptr64;
  do {
    assumed = old;
    value.i2.x = assumed.i2.x - sub.x;
    value.i2.y = assumed.i2.y - sub.y;
    old.ull = atomicCAS(ptr64, assumed.ull, value.ull);
  } while (old.ull != assumed.ull);
}

// GetAdaptiveBeam is used by ExpandArc and FinalizeProcessNonemitting
//
// Given the fact that the token queues are too small to store
// all possible tokens in the worst case scenario (where we could generate
// "nstates" tokens),
// we need to tighten the beam if we notice that we are at risk of overflowing
// either the aux_q
// or the main_q

__device__ void UpdateAdaptiveBeam(const DeviceParams &cst_dev_params,
                                   const int aux_q_index_block_offset,
                                   IntegerCostType min_int_cost,
                                   int2 *adaptive_int_beam_with_validity_index,
                                   LaneCounters *lane_counters) {
  int32 beam_valid_until_idx = adaptive_int_beam_with_validity_index->y;
  if (aux_q_index_block_offset < beam_valid_until_idx) return;  // nothing to do

  CostType beam = orderedIntToFloat(adaptive_int_beam_with_validity_index->x);
  while (aux_q_index_block_offset >= beam_valid_until_idx) {
    beam /= 2;
    beam_valid_until_idx += cst_dev_params.adaptive_beam_bin_width;
  }
  // FIXME can overflow
  IntegerCostType new_int_cutoff =
      floatToOrderedInt(orderedIntToFloat(min_int_cost) + beam);
  IntegerCostType int_beam = floatToOrderedInt(beam);
  adaptive_int_beam_with_validity_index->x = int_beam;
  adaptive_int_beam_with_validity_index->y = beam_valid_until_idx;
  // We can have races between the two atomics
  // However the worst than can happen is a CTA might delay updating the beam
  // This is not a critical bug. However, once we have a floatToOrderedInt
  // that is generating unsigned ints, we could merge the two atomics into a
  // single atomic64
  atomicMin(&lane_counters->adaptive_int_beam_with_validity_index.x, int_beam);
  atomicMax(&lane_counters->adaptive_int_beam_with_validity_index.y,
            beam_valid_until_idx);
  atomicMin(&lane_counters->int_cutoff, new_int_cutoff);
}

//
// HASHMAP
//

__device__ __forceinline__ int hash_func(int key) {
  return key;  // using identity for now
}

__device__ __forceinline__ HashmapValueT hashmap_find(
    HashmapValueT *d_map_values, int key, int capacity, int *hash_idx2val) {
  int hash_idx = hash_func(key) % capacity;
  int c = 0;

  do {
    HashmapValueT val = d_map_values[hash_idx];
    if (val.key == key) {
      *hash_idx2val = hash_idx;
      return val;
    }
    if (val.key == KALDI_CUDA_DECODER_HASHMAP_NO_KEY) break;

    hash_idx = (hash_idx + 1) % capacity;
    ++c;
  } while (c < capacity);

  *hash_idx2val = NULL;
  return {KALDI_CUDA_DECODER_HASHMAP_NO_KEY,
          0,
          {INT_MAX, -1}};  // no such key in map
}

__device__ void hashmap_insert(HashmapValueT *d_map_values, int key,
                               int int_cost, int arg, int capacity,
                               int *local_idx) {
  int hash_idx = hash_func(key) % capacity;
  int c = 0;
  HashmapValueT *d_val = NULL;
  do {
    d_val = &d_map_values[hash_idx];
    int old = atomicCAS(&d_val->key, KALDI_CUDA_DECODER_HASHMAP_NO_KEY, key);
    if (old == KALDI_CUDA_DECODER_HASHMAP_NO_KEY || old == key)
      break;  // found a spot
    hash_idx = (hash_idx + 1) % capacity;
    ++c;
  } while (c < capacity);
  if (!d_val) return;

  // Updating values
  *local_idx = atomicAdd(&d_val->count, 1);
  atomicMinI2(&d_val->min_and_argmin_int_cost, {int_cost, arg});
}
}  // end namespace CudaDecode
}  // end namespace kaldi

#endif
