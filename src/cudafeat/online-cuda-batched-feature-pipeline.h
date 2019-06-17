// cudafeat/online-cuda-feature-pipeline.h

// Copyright 2013-2014   Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_CUDAFEAT_ONLINE_CUDA_BATCHED_FEATURE_PIPELINE_H_
#define KALDI_CUDAFEAT_ONLINE_CUDA_BATCHED_FEATURE_PIPELINE_H_

#include <deque>
#include <string>
#include <vector>

#include "base/kaldi-error.h"
#include "cudafeat/batched-feature-mfcc-cuda.h"
#include "matrix/matrix-lib.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "util/common-utils.h"

namespace kaldi {

class OnlineCudaBatchedFeaturePipeline {
 public:
  explicit OnlineCudaBatchedFeaturePipeline(
      const OnlineNnet2FeaturePipelineConfig &config, int32 batch_size_,
      int32 max_samples_chunk_, int32 max_states_, int32 window_size_,
      int32 feat_dim_);

  void ComputeFeatures(const CuMatrix<BaseFloat> &cu_wave,
                       BaseFloat sample_freq,
                       CuMatrix<BaseFloat> *input_features,
                       CuMatrix<BaseFloat> *ivector_features,
                       CuMatrix<BaseFloat> *cmvn_feats,
                       const std::vector<Channels> &channels,
                       const std::vector<Channels> &starts,
                       const std::vector<Channels> &end,
                       std::vector<Channels> &sample_valid,
                       std::vector<Channels> *n_frames_valid);

  void InitChannels(const std::vector<Channels> &channel,
                    const std::vector<Channels> &starts,
                    const std::vector<Channels> &end,
                    const std::vector<int> &n_frames_valid);

  void ComputeFrameValidNumber(const int partial_batch_size,
                               const std::vector<Channels> &starts,
                               const std::vector<Channels> &end,
                               std::vector<Channels> &sample_valid,
                               const FrameExtractionOptions &frame_opts,
                               std::vector<int> *n_frames_valid);

  void ResetChannel(int batch_partial_size);

  ~OnlineCudaBatchedFeaturePipeline();

 private:
  OnlineNnet2FeaturePipelineInfo info_;
  BatchedCudaMfcc *mfcc;

  // batch size
  int32 batch_size;

  // device pointer
  KernelParam *kernel_params;

  // frame valid

  int32 max_samples_chunks;
  int32 cache_num_samples;
};
}  // namespace kaldi

#endif  // KALDI_CUDAFEAT_ONLINE_CUDA_FEATURE_EXTRACTOR_H_
