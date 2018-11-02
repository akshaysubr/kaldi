// nnet3/decodable-online-looped.cc

// Copyright  2017  Johns Hopkins University (author: Daniel Povey)

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

#include <nnet3/decodable-online-looped.h>
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

DecodableAmNnetLoopedOnlineCuda::DecodableAmNnetLoopedOnlineCuda(
      const DecodableNnetSimpleLoopedInfo &info,
      OnlineFeatureInterface *input_features,
      OnlineFeatureInterface *ivector_features):
      DecodableNnetLoopedOnlineBase(info, input_features, ivector_features) {
 };

DecodableAmNnetLoopedOnlineCuda::~DecodableAmNnetLoopedOnlineCuda() {
}

BaseFloat* DecodableAmNnetLoopedOnlineCuda::GetNnet3Output(int32 subsampled_frame) {
  EnsureFrameIsComputed(subsampled_frame);
  cudaStreamSynchronize(cudaStreamPerThread);      
  
  BaseFloat *frame_nnet3_out = current_log_post_.Data()+(subsampled_frame-current_log_post_subsampled_offset_)*current_log_post_.Stride();
  return frame_nnet3_out;
}

} // namespace nnet3
} // namespace kaldi
