// cudafeat/online-cuda-feature-pipleine.cc

// Copyright    2013  Johns Hopkins University (author: Daniel Povey)

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

#include "cudafeat/online-cuda-batched-feature-pipeline.h"

namespace kaldi {

OnlineCudaBatchedFeaturePipeline::OnlineCudaBatchedFeaturePipeline(
    const OnlineNnet2FeaturePipelineConfig &config, int32 batch_size_,
    int32 max_samples_chunk_, int32 max_states_, int32 window_size_,
    int32 feat_dim_)
    : info_(config),
      mfcc(NULL),
      batch_size(batch_size_),
      max_samples_chunks(max_samples_chunk_) {
  if (info_.feature_type == "mfcc") {
    mfcc = new BatchedCudaMfcc(info_.mfcc_opts, batch_size, max_samples_chunk_,
                               max_states_);
  }

  // have mfcc states here
  int frame_nums =
      NumFrames(max_samples_chunks, info_.mfcc_opts.frame_opts, true);
  int frame_shift = info_.mfcc_opts.frame_opts.WindowShift();

  cache_num_samples = max_samples_chunks - frame_shift * frame_nums;

  if (info_.use_ivectors) {
    OnlineIvectorExtractionConfig ivector_extraction_opts;
    ReadConfigFromFile(config.ivector_extraction_config,
                       &ivector_extraction_opts);
    info_.ivector_extractor_info.Init(ivector_extraction_opts);

    // Only these ivector options are currently supported
    ivector_extraction_opts.use_most_recent_ivector = true;
    ivector_extraction_opts.greedy_ivector_extractor = true;
  }
  // preallocate
  kernel_params = static_cast<KernelParam *>(
      CuDevice::Instantiate().Malloc(sizeof(KernelParam) * batch_size));
}

OnlineCudaBatchedFeaturePipeline::~OnlineCudaBatchedFeaturePipeline() {
  // deallocate
  CuDevice::Instantiate().Free(kernel_params);

  if (mfcc != NULL) delete mfcc;
}

void OnlineCudaBatchedFeaturePipeline::ComputeFrameValidNumber(
    const int partial_batch_size, const std::vector<Channels> &starts,
    const std::vector<Channels> &end, std::vector<Channels> &sample_valid,
    const FrameExtractionOptions &frame_opts,
    std::vector<int> *n_frames_valid) {
  KALDI_ASSERT(n_frames_valid->size() == sample_valid.size());
  // populate frame_valid vector
  for (int i = 0; i < partial_batch_size; i++) {
    int32 num_samples;
    // num_samples
    if (starts[i] == 1 && end[i] == 1) {
      // both start and end
      num_samples = sample_valid[i];
    } else if (starts[i] == 1) {
      // only start
      num_samples = sample_valid[i];
    }
    // middle chunk or end but not start
    else {
      num_samples = sample_valid[i] + cache_num_samples;
    }

    // calc num frames with cpu func
    int32 num_frames = NumFrames(num_samples, frame_opts, true);
    (*n_frames_valid)[i] = num_frames;
  }
}

// channel, starts, end in host
void OnlineCudaBatchedFeaturePipeline::InitChannels(
    const std::vector<Channels> &channel, const std::vector<Channels> &starts,
    const std::vector<Channels> &end, const std::vector<int> &n_frames_valid) {
  // init channel and store it into vector
  int batch_partial_size = channel.size();
  std::vector<KernelParam> kernel_vector;

  for (int i = 0; i < batch_partial_size; i++) {
    // populate the vector
    KernelParam param;
    param.compute_channel = channel[i];
    param.start_utt = starts[i];
    param.end_utt = end[i];
    param.n_frames_valid = n_frames_valid[i];
    kernel_vector.push_back(param);
  }
  // now copy to the host
  CU_SAFE_CALL(cudaMemcpyAsync(kernel_params, &kernel_vector[0],
                               batch_partial_size * sizeof(KernelParam),
                               cudaMemcpyHostToDevice, cudaStreamPerThread));
}

void OnlineCudaBatchedFeaturePipeline::ResetChannel(int batch_partial_size) {
  // reset channels for mfcc and i-vector
  mfcc->ResetChannel(kernel_params, batch_partial_size);
}

void OnlineCudaBatchedFeaturePipeline::ComputeFeatures(
    const CuMatrix<BaseFloat> &cu_wave, BaseFloat sample_freq,
    CuMatrix<BaseFloat> *input_features, CuMatrix<BaseFloat> *ivector_features,
    CuMatrix<BaseFloat> *cmvn_feats, const std::vector<Channels> &channels,
    const std::vector<Channels> &starts, const std::vector<Channels> &end,
    std::vector<Channels> &sample_valid,
    std::vector<Channels> *n_frames_valid) {
  KALDI_ASSERT(ivector_features == NULL);
  KALDI_ASSERT(cmvn_feats == NULL);
  n_frames_valid->resize(sample_valid.size());
  int batch_partial_size = channels.size();
  // compute frame valid number vector
  ComputeFrameValidNumber(batch_partial_size, starts, end, sample_valid,
                          info_.mfcc_opts.frame_opts, n_frames_valid);

  // init channel first
  InitChannels(channels, starts, end, *n_frames_valid);

  // mfcc features, fbank will be added
  if (info_.feature_type == "mfcc") {
    // MFCC
    float vtln_warp = 1.0;
    mfcc->ComputeBatchedFeatures(cu_wave, sample_freq, vtln_warp,
                                 input_features, kernel_params,
                                 batch_partial_size);
  } else {
    KALDI_ASSERT(false);
  }

  // reset channel
  ResetChannel(batch_partial_size);
}

}  // namespace kaldi
