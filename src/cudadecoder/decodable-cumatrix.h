// cudadecoder/decodable-cumatrix.h
/*
 * Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef KALDI_CUMATRIX_H_
#define KALDI_CUMATRIX_H_

#include "cudadecoder/cuda-decodable-itf.h"
#include "cudamatrix/cu-matrix.h"
#include "decoder/decodable-matrix.h"

namespace kaldi {
/**
  Cuda Decodable matrix.  Takes transition model and posteriors and provides
  an interface similar to the Decodable Interface
  */
class DecodableCuMatrixMapped : public CudaDecodableInterface {
public:
  // This constructor creates an object that will not delete "likes" when done.
  // the frame_offset is the frame the row 0 of 'likes' corresponds to, would be
  // greater than one if this is not the first chunk of likelihoods.
  DecodableCuMatrixMapped(const TransitionModel &tm,
                          const CuMatrixBase<BaseFloat> &likes,
                          int32 frame_offset = 0);

  virtual int32 NumFramesReady() const;

  virtual bool IsLastFrame(int32 frame) const;

  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    KALDI_ASSERT(false);
  };

  // Note: these indices are 1-based.
  virtual int32 NumIndices() const;

  virtual ~DecodableCuMatrixMapped(){};

  // returns cuda pointer to nnet3 output
  virtual BaseFloat *GetLogLikelihoodsCudaPointer(int32 subsampled_frame);

private:
  const TransitionModel &trans_model_; // for tid to pdf mapping
  const CuMatrixBase<BaseFloat> *likes_;

  int32 frame_offset_;

  // raw_data and stride_ are a kind of fast look-aside for 'likes_', to be
  // used when KALDI_PARANOID is false.
  const BaseFloat *raw_data_;
  int32 stride_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableCuMatrixMapped);
};
}

#endif
