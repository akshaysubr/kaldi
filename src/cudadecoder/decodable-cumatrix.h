// cudadecoder/decodable-cumatrix.h
/*
 * Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 * Authors:  Hugo Braun, Justin Luitjens, Ryan Leary
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef KALDI_CUDA_DECODER_DECODABLE_CUMATRIX_H_
#define KALDI_CUDA_DECODER_DECODABLE_CUMATRIX_H_

#include "cudamatrix/cu-matrix.h"
#include "decoder/decodable-matrix.h"

namespace kaldi {
namespace cuda_decoder {

/**
  Cuda Decodable matrix.  Takes transition model and posteriors and provides
  an interface similar to the Decodable Interface
  */
class DecodableCuMatrixMapped {
 public:
  // This constructor creates an object that will not delete "likes" when done.
  // the frame_offset is the frame the row 0 of 'likes' corresponds to, would be
  // greater than one if this is not the first chunk of likelihoods.
  DecodableCuMatrixMapped(const TransitionModel &tm,
                          const CuSubMatrix<BaseFloat> &likes,
                          int32 frame_offset = 0);

  int32 NumFramesReady() const;

  // Note: these indices are 1-based.
  int32 NumIndices() const;

  virtual ~DecodableCuMatrixMapped(){};

  // returns cuda pointer to nnet3 output
  BaseFloat *GetLogLikelihoodsCudaPointer(int32 subsampled_frame);

  void SetFrameOffset(int32 frame_offset) { frame_offset_ = frame_offset; }
  void SetValidNRowsInMatrix(int32 n_valid_likes) {
    n_valid_likes_ = n_valid_likes;
  }

 private:
  const TransitionModel &trans_model_;  // for tid to pdf mapping
  CuSubMatrix<BaseFloat> likes_;
  int32 n_valid_likes_;
  int32 frame_offset_;
};

}  // end namespace cuda_decoder
}  // end namespace kaldi.

#endif  // KALDI_CUDA_DECODER_DECODABLE_CUMATRIX_H_
