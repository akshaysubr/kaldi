// cudadecoder/batched-static-nnet3.h
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Hugo Braun
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

#if HAVE_CUDA == 1

#ifndef KALDI_CUDA_DECODER_BATCHED_STATIC_NNET3_H_
#define KALDI_CUDA_DECODER_BATCHED_STATIC_NNET3_H_

// Following define is NOT an upper bound for max_batch_size
// It only concerns the nnet3 compiled computation
// If we use a batch size > MAX_COMPUTE_BATCH_SIZE, we will run nnet3
// multiple times, each computing minibatches of size MAX_COMPUTE_BATCH_SIZE
// MAX_COMPUTE_BATCH_SIZE is defined to be big enough to hide kernel launch
// latency and increase the arithmetic intensity of the GEMMs
// not not bigger so that running partial batches is faster
// (e.g. running a batch size = 72 with max_batch_size_=512)
#define MAX_COMPUTE_BATCH_SIZE 64

#include "cudadecoder/batched-static-nnet3-kernels.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"

namespace kaldi {
namespace cuda_decoder {

struct BatchedStaticNnet3Config {
  BatchedStaticNnet3Config()
      : max_batch_size(200), nchannels(-1), has_ivector(false) {}
  nnet3::NnetSimpleComputationOptions compute_opts;
  int max_batch_size;
  int nchannels;
  bool has_ivector;
};

class BatchedStaticNnet3 {
 public:
  BatchedStaticNnet3(const BatchedStaticNnet3Config &config,
                     const nnet3::AmNnetSimple &am_nnet)
      : config_(config),
        am_nnet_(am_nnet),
        max_batch_size_(config.max_batch_size),
        has_ivector_(config.has_ivector),
        log_priors_(am_nnet.Priors()) {
    nchannels_ = (config.nchannels != -1) ? config.nchannels : max_batch_size_;
    KALDI_ASSERT(max_batch_size_ > 0);
    nnet3_batch_size_ = std::min(max_batch_size_, MAX_COMPUTE_BATCH_SIZE);
    KALDI_ASSERT(nchannels_ >= max_batch_size_);
    ReadParametersFromModelAndConfig();
    CompileNnet3();
    Allocate();
  }

  virtual ~BatchedStaticNnet3() { Deallocate(); }

  void RunBatch(const std::vector<int> &channels,
                const std::vector<BaseFloat *> &d_features,
                const int features_stride,
                const std::vector<BaseFloat *> &d_ivectors,
                const std::vector<int> &n_input_frames_valid,
                const std::vector<bool> &is_last_chunk,
                CuMatrix<BaseFloat> *d_all_log_posteriors,
                std::vector<std::vector<std::pair<int, BaseFloat *>>>
                    *all_frames_log_posteriors);

  void FormatOutputPtrs(
      const std::vector<int> &channels,
      CuMatrix<BaseFloat> *d_all_log_posteriors,
      std::vector<std::vector<std::pair<int, BaseFloat *>>>
          *all_frames_log_posteriors_ptrs,
      const std::vector<int> &n_output_frames_valid,
      const std::vector<int> *n_output_frames_valid_offset = NULL);

  void InitChannel(int32 ichannel) {
    KALDI_ASSERT(ichannel < nchannels_);
    channel_n_frames_in_context_[ichannel] = 0;
  }

  int GetNOutputFramesPerChunk() { return output_frames_per_chunk_; }
  int GetTotalNnet3RightContext() { return total_nnet_right_context_; }

 private:
  // Compiling nnet3 using that computation request
  void ReadParametersFromModelAndConfig();
  // Define the computation request for nnet3 based on parameters
  void SetComputationRequest();
  void Allocate();
  void PresetKernelParams();
  void Deallocate();
  void CompileNnet3();
  void RunNnet3(CuMatrix<BaseFloat> *d_all_log_posteriors, int batch_size);
  void BatchContextSwitch(const std::vector<int> &channels,
                          const std::vector<BaseFloat *> &d_features,
                          const int features_stride,
                          const std::vector<BaseFloat *> &d_ivectors,
                          const std::vector<int> &n_input_frames_valid,
                          bool flush_eos_context,
                          std::vector<int> *n_output_frames_valid);

  BatchedStaticNnet3Config config_;
  cudaStream_t st_;
  nnet3::AmNnetSimple am_nnet_;
  int max_batch_size_;
  int nnet3_batch_size_;
  int nchannels_;
  bool has_ivector_;
  CuVector<BaseFloat> log_priors_;

  // Extracted from config or models
  int input_dim_;    // mfcc dim
  int ivector_dim_;  // ivector dim
  int input_frames_per_chunk_;
  int input_frames_per_chunk_with_context_;
  int total_nnet_left_context_;
  int total_nnet_right_context_;
  int total_nnet_context_;
  int output_frames_per_chunk_;
  int subsampling_factor_;

  // Storing frames which will be used in future context
  // If the channel has just been resetted, those frames are empty.
  // Otherwise, it contains at most total_nnet_context_ frames
  CuMatrix<BaseFloat> d_all_context_frames_;
  CuMatrix<BaseFloat> d_batch_with_context_;
  CuMatrix<BaseFloat> d_nnet3_input_;
  CuMatrix<BaseFloat> d_nnet3_ivectors_;
  CuMatrix<BaseFloat> d_nnet3_output_;
  CuMatrix<BaseFloat> d_batch_ivectors_;
  CuMatrix<BaseFloat> d_all_log_posteriors_;
  CuMatrix<BaseFloat> d_all_eos_log_posteriors_;
  // batch slot assignement. Size [max_batch_size]
  BatchSlotAssignment *d_batch_slot_assignement_;
  BatchSlotAssignment *h_batch_slot_assignement_;
  BatchedStaticNnet3KernelParams context_switch_kernel_params_;
  cudaEvent_t batch_slot_assignement_copy_evt_;
  // Number of frames already stored in context
  // Size [nchannels]
  // If channel not initialized, equals to -1
  std::vector<int> channel_n_frames_in_context_;
  std::vector<int> n_output_frames_valid_;

  // Used to flush context at eos
  std::vector<int> eos_channels_;
  std::vector<BaseFloat *> d_eos_features_;
  std::vector<BaseFloat *> d_eos_ivectors_;
  std::vector<int> eos_n_input_frames_valid_;
  std::vector<int> eos_n_output_frames_valid_;
  std::vector<int> eos_n_output_frames_offset_;

  std::unique_ptr<nnet3::CachingOptimizingCompiler> compiler_;
  std::shared_ptr<const nnet3::NnetComputation>
      computation_;  // shared because returned as shared by compiler
  nnet3::ComputationRequest request_;
};
}  // cuda_decoder
}  // kaldi

#endif  // KALDI_CUDA_DECODER_BATCHED_STATIC_NNET3_H_
#endif  // HAVE_CUDA
