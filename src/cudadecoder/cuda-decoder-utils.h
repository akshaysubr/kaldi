// cudadecoder/cuda-decoder-utils.h
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

#ifndef KALDI_DECODER_CUDA_DECODER_UTILS_H_
#define KALDI_DECODER_CUDA_DECODER_UTILS_H_
#include <cuda_runtime_api.h>
#include <string>
#include "util/stl-utils.h"
#include "cudamatrix/cu-device.h"

#define KALDI_CUDA_DECODER_DIV_ROUND_UP(a, b) ((a + b - 1) / b)

#define CUDA_DECODER_ASSERT(val, recoverable)                               \
  {                                                                         \
    if ((val) != true) {                                                    \
      throw CudaDecoderException("CUDA_DECODER_ASSERT", __FILE__, __LINE__, \
                                 recoverable)                               \
    }                                                                       \
  }
// Macro for checking cuda errors following a cuda launch or api call
#define KALDI_DECODER_CUDA_CHECK_ERROR()                                  \
  {                                                                       \
    cudaError_t e = cudaGetLastError();                                   \
    if (e != cudaSuccess) {                                               \
      throw CudaDecoderException(cudaGetErrorName(e), __FILE__, __LINE__, \
                                 false);                                  \
    }                                                                     \
  }

#define KALDI_DECODER_CUDA_API_CHECK_ERROR(e)                             \
  {                                                                       \
    if (e != cudaSuccess) {                                               \
      throw CudaDecoderException(cudaGetErrorName(e), __FILE__, __LINE__, \
                                 false);                                  \
    }                                                                     \
  }

#define KALDI_CUDA_DECODER_1D_KERNEL_LOOP(i, n)                \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define KALDI_CUDA_DECODER_1D_BLOCK_OFFSET_KERNEL_LOOP(offset, th_idx, n) \
  for (int offset = blockIdx.x * blockDim.x, th_idx = threadIdx.x;        \
       offset < (n); offset += blockDim.x * gridDim.x)

#define KALDI_CUDA_DECODER_IS_LAST_1D_THREAD() (threadIdx.x == (blockDim.x - 1))

#define KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.y; i < (n); i += gridDim.y)

#define KALDI_CUDA_DECODER_DIV_ROUND_UP(a, b) ((a + b - 1) / b)

#define KALDI_CUDA_DECODER_1D_BLOCK 256
#define KALDI_CUDA_DECODER_LARGEST_1D_BLOCK 1024
#define KALDI_CUDA_DECODER_ONE_THREAD_BLOCK 1

inline dim3 KALDI_CUDA_DECODER_NUM_BLOCKS(int N, int M) {
  dim3 grid;
  // TODO MAX_NUM_BLOCKS
  grid.x = KALDI_CUDA_DECODER_DIV_ROUND_UP(N, KALDI_CUDA_DECODER_1D_BLOCK);
  grid.y = M;
  return grid;
}

#include <cuda.h>
#include "fst/fstlib.h"
#include "nnet3/nnet-utils.h"
#include "util/stl-utils.h"

namespace kaldi {
namespace CudaDecoder {
typedef float CostType;
// IntegerCostType is the type used in the lookup table d_state_best_cost
// and the d_cutoff
// We use a 1:1 conversion between CostType <--> IntegerCostType
// IntegerCostType is used because it triggers native atomic operations
// (CostType does not)
typedef int32 IntegerCostType;
typedef int32 LaneId;
typedef int32 ChannelId;

template <typename T>
// if necessary, make a version that always use ld_ as the next power of 2
class DeviceMatrix {
  T *data_;
  void Allocate() {
    KALDI_ASSERT(nrows_ > 0);
    KALDI_ASSERT(ld_ > 0);
    KALDI_ASSERT(!data_);
    data_ = static_cast<T *>(
        CuDevice::Instantiate().Malloc((size_t)nrows_ * ld_ * sizeof(*data_)));
    KALDI_ASSERT(data_);
  }
  void Free() {
    KALDI_ASSERT(data_);
    CuDevice::Instantiate().Free(data_);
  }

 protected:
 int32 ld_;     // leading dimension
  int32 nrows_;  // leading dimension
 public:
  DeviceMatrix() : data_(NULL), ld_(0), nrows_(0) {}

  virtual ~DeviceMatrix() {
    if (data_) Free();
  }

  void Resize(int32 nrows, int32 ld) {
    KALDI_ASSERT(nrows > 0);
    KALDI_ASSERT(ld > 0);
    nrows_ = nrows;
    ld_ = ld;
  }

  T *MutableData() {
    if (!data_) Allocate();
    return data_;
  }
  // abstract getInterface...
};

// The interfaces contains CUDA code
// We will declare them in a .cu file
template <typename T>
class LaneMatrixInterface;
template <typename T>
class ChannelMatrixInterface;
template <typename T>

class DeviceLaneMatrix : public DeviceMatrix<T> {
 public:
  LaneMatrixInterface<T> GetInterface() {
    return {this->MutableData(), this->ld_};
  }

  T *lane(const int32 ilane) { return &this->MutableData()[ilane * this->ld_]; }
};

template <typename T>
class DeviceChannelMatrix : public DeviceMatrix<T> {
 public:
  ChannelMatrixInterface<T> GetInterface() {
    return {this->MutableData(), this->ld_};
  }
  T *channel(const int32 ichannel) {
    return &this->MutableData()[ichannel * this->ld_];
  }
};

struct LaneCounters {
  // Contains both main_q_end and narcs
  // End index of the main queue
  // only tokens at index i with i < main_q_end
  // are valid tokens
  // Each valid token the subqueue main_q[main_q_offset, main_q_end[ has
  // a number of outgoing arcs (out-degree)
  // main_q_narcs is the sum of those numbers
  //
  // We sometime need to update both end and narcs at the same time,
  // which is why they're packed together
  int2 main_q_narcs_and_end;
  // contains the requested queue length which can
  // be larger then the actual queue length in the case of overflow
  int32 main_q_requested;
  int32 aux_q_requested;

  // Some kernels need to perform some operations before exiting
  // n_CTA_done is a counter that we increment when a CTA (CUDA blocks)
  // is done
  // Each CTA then tests the value for n_CTA_done to detect if it's the last to
  // exit
  // If that's the cast, it does what it has to do, and sets n_CTA_done back to
  // 0
  int32 aux_q_end;
  int32 post_expand_aux_q_end;  // used for double buffering
  int32 main_q_n_extra_prev_tokens;

  // Depending on the value of the parameter "max_tokens_per_frame"
  // we can end up with an overflow when generating the tokens for a frame
  // We try to prevent this from happening using an adaptive beam
  // if an overflow happens, then the kernels no longer insert any data into
  // the queues and set overflow flag to true.
  // queue length.
  // Even if that flag is set, we can continue the execution (quality
  // of the output can be lowered)
  // We use that flag to display a warning to stderr
  int32 q_overflow;

  // ExpandArcs does not use at its input the complete main queue
  // It only reads from the index range [main_q_local_offset, end[
  int32 main_q_local_offset;
  int32 main_q_global_offset;
  int32 main_q_extra_prev_tokens_global_offset;

  IntegerCostType min_int_cost;
  IntegerCostType int_beam;
  int2 adaptive_int_beam_with_validity_index;

  IntegerCostType
      int_cutoff;  // min_cost + beam (if min_cost < INF, otherwise INF)

  // Only valid after calling GetBestCost
  int2 min_int_cost_and_arg;
  int32 nfinals;
  int32 has_reached_final;
};

//
// Parameters used by a decoder channel
// Their job is to save the state of the decoding
// channel between frames
//
struct ChannelCounters {
  // Cutoff for the current frame
  // Contains both the global min cost (min cost for that frame)
  // And the current beam
  // We use an adaptive beam, so the beam might change during computation
  CostType prev_beam;

  // main_q_end and main_q_narcs at the end of the previous frame
  int2 prev_main_q_narcs_and_end;
  int32 prev_main_q_n_extra_prev_tokens;

  // The token at index i in the main queue has in reality
  // a global index of (i + main_q_global_offset)
  // This global index is unique and takes into account that
  // we've flushed the main_q back to the host. We need unique indexes
  // for each token in order to have valid token.prev_token data members
  // and be able to backtrack at the end
  int32 prev_main_q_global_offset;
  int32 prev_main_q_extra_prev_tokens_global_offset;

  // Only valid after calling GetBestCost
  // different than min_int_cost : we include the "final" cost
  int2 min_int_cost_and_arg_with_final;
  int2 min_int_cost_and_arg_without_final;
};
//
// Data structures used by the kernels
//

// Count of tokens and arcs in a queue
// narcs = sum(number of arcs going out of token i next state) for each token
// in the queue
// We use this struct to keep the two int32s adjacent in memory
// we need this in order to update both using an atomic64 operation
struct TokenAndArcCount {
  int32 ntokens;
  int32 narcs;
};

// Union structure of the TokenAndArcCount
// We use split to access the int32s
// We use both to update both using an atomic64
union TokenAndArcCountUnion {
  TokenAndArcCount split;
  unsigned long long both;
};

//
// Used for the cutoff
// cutoff = min_cost + beam
// We store both separatly because we have an adaptive beam
// We may change the beam after discovering min_cost
// we need to keep track of min_cost to apply the new beam
// (we don't know what the old beam was)
//
// Native float and Integers version
//
struct MinCostAndBeam {
  CostType min_cost;
  CostType beam;
};

struct MinCostAndBeamIntegers {
  IntegerCostType min_cost;
  IntegerCostType beam;
};
class CudaDecoderException : public std::exception {
 public:
  CudaDecoderException(const char *str_, const char *file_, int line_,
                       const bool recoverable_)
      : str(str_),
        file(file_),
        line(line_),
        buffer(std::string(file) + ":" + std::to_string(line) + " :" +
               std::string(str)),
        recoverable(recoverable_) {}
  const char *what() const throw() { return buffer.c_str(); }

  const char *str;
  const char *file;
  const int line;
  const std::string buffer;
  const bool recoverable;
};

// InfoToken contains data that needs to be saved for the backtrack
// in GetBestPath
// It will be moved back to CPU memory using a InfoTokenVector
struct __align__(8) InfoToken {
  int32 prev_token;
  int32 arc_idx;
  __host__ bool IsUniqueTokenForStateAndFrame() {
    // This is a trick used to save space and PCI-E bandwidth (cf
    // preprocess_in_place kernel)
    // This token is associated with a (next) state s, created during the
    // processing of frame f.
    // If we have multiple tokens associated with the state s in the frame f,
    // arc_idx < 0 and -arc_idx is the
    // count of such tokens. We will then have to look at another list to read
    // the actually arc_idx and prev_token values
    // If the current token is the only one, prev_token and arc_idx are valid
    // and can be used directly
    return (arc_idx >= 0);
  }

  // Called if this token is linked to others tokens in the same frame (cf
  // comments for IsUniqueTokenForStateAndFrame)
  // return the {offset,size} pair necessary to list those tokens in the
  // extra_prev_tokens list
  // They are stored at offset "offset", and we have "size" of those
  __host__ std::pair<int32, int32> GetNextStateTokensList() {
    KALDI_ASSERT(!IsUniqueTokenForStateAndFrame());

    return {prev_token, -arc_idx};
  }
};

//
// InfoTokenVector
// Vector for InfoToken that uses CPU pinned memory
// We use it to transfer the relevant parts of the tokens
// back to the CPU memory
//
class InfoTokenVector {
  int32 capacity_, size_;
  // Stream used for the async copies device->host
  cudaStream_t copy_st_;
  InfoToken *h_data_;

 public:
  InfoTokenVector(int32 initial_capacity, cudaStream_t copy_st_);
  InfoTokenVector(const InfoTokenVector &other);  // TODO refactor
  void Clone(const InfoTokenVector &other);
  void Reset();
  void CopyFromDevice(InfoToken *d_ptr, int32 count);
  int32 Size() const { return size_; }
  void Reserve(int32 min_capacity);
  InfoToken *GetRawPointer() const;
  virtual ~InfoTokenVector();
};

//
// Hashmap
//

}  // end namespace CudaDecoder
}  // end namespace kaldi

#endif
