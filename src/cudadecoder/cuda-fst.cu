// cudadecoder/cuda-fst.cu
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

#include <cuda_runtime_api.h>
#include <nvToolsExt.h>
#include "cudadecoder/cuda-decoder-utils.h"
#include "cudadecoder/cuda-fst.h"

namespace kaldi {
namespace CudaDecoder {
void CudaFst::Initialize(const fst::Fst<StdArc> &fst,
                         const TransitionModel &trans_model) {
  nvtxRangePushA("CudaFst constructor");
  // count states since Fst doesn't provide this functionality
  num_states_ = 0;
  for (fst::StateIterator<fst::Fst<StdArc> > iter(fst); !iter.Done();
       iter.Next())
    ++num_states_;

  start_ = fst.Start();

  // allocate and initialize offset arrays
  h_final_.resize(num_states_);
  h_e_offsets_.resize(num_states_ + 1);
  h_ne_offsets_.resize(num_states_ + 1);

  d_e_offsets_ = static_cast<unsigned int *>(CuDevice::Instantiate().Malloc(
      (num_states_ + 1) * sizeof(*d_e_offsets_)));
  d_ne_offsets_ = static_cast<unsigned int *>(CuDevice::Instantiate().Malloc(
      (num_states_ + 1) * sizeof(*d_ne_offsets_)));
  d_final_ = static_cast<float *>(
      CuDevice::Instantiate().Malloc((num_states_) * sizeof(*d_final_)));
  KALDI_ASSERT(d_e_offsets_);
  KALDI_ASSERT(d_ne_offsets_);
  KALDI_ASSERT(d_final_);

  // iterate through states and arcs and count number of arcs per state
  e_count_ = 0;
  ne_count_ = 0;

  // Init first offsets
  h_ne_offsets_[0] = 0;
  h_e_offsets_[0] = 0;
  for (int i = 0; i < num_states_; i++) {
    h_final_[i] = fst.Final(i).Value();
    // count emiting and non_emitting arcs
    for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, i); !aiter.Done();
         aiter.Next()) {
      StdArc arc = aiter.Value();
      int32 ilabel = arc.ilabel;
      if (ilabel != 0) {  // emitting
        e_count_++;
      } else {  // non-emitting
        ne_count_++;
      }
    }
    h_ne_offsets_[i + 1] = ne_count_;
    h_e_offsets_[i + 1] = e_count_;
  }

  // We put the emitting arcs before the nonemitting arcs in the arc list
  // adding offset to the non emitting arcs
  // we go to num_states_+1 to take into account the last offset
  for (int i = 0; i < num_states_ + 1; i++)
    h_ne_offsets_[i] += e_count_;  // e_arcs before

  arc_count_ = e_count_ + ne_count_;

  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpy(
      d_e_offsets_, &h_e_offsets_[0], (num_states_ + 1) * sizeof(*d_e_offsets_),
      cudaMemcpyHostToDevice));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpy(
      d_ne_offsets_, &h_ne_offsets_[0],
      (num_states_ + 1) * sizeof(*d_ne_offsets_), cudaMemcpyHostToDevice));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpy(d_final_, &h_final_[0],
                                                num_states_ * sizeof(*d_final_),
                                                cudaMemcpyHostToDevice));

  h_arc_weights_.resize(arc_count_);
  h_arc_nextstates_.resize(arc_count_);
  // ilabels (nnet3 indexing)
  std::vector<int32> h_arc_pdf_ilabels_(arc_count_);
  // ilabels (fst indexing)
  h_arc_id_ilabels_.resize(arc_count_);
  h_arc_olabels_.resize(arc_count_);

  d_arc_weights_ = static_cast<float *>(
      CuDevice::Instantiate().Malloc(arc_count_ * sizeof(*d_arc_weights_)));
  d_arc_nextstates_ = static_cast<StateId *>(
      CuDevice::Instantiate().Malloc(arc_count_ * sizeof(*d_arc_nextstates_)));

  // Only the ilabels for the e_arc are needed on the device
  d_arc_pdf_ilabels_ = static_cast<int32 *>(
      CuDevice::Instantiate().Malloc(e_count_ * sizeof(*d_arc_pdf_ilabels_)));

  KALDI_ASSERT(d_arc_weights_);
  KALDI_ASSERT(d_arc_nextstates_);
  KALDI_ASSERT(d_arc_pdf_ilabels_);

  // We do not need the olabels on the device - GetBestPath is on CPU

  // now populate arc data
  int e_idx = 0;
  int ne_idx = e_count_;  // starts where e_offsets_ ends
  for (int i = 0; i < num_states_; i++) {
    for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, i); !aiter.Done();
         aiter.Next()) {
      StdArc arc = aiter.Value();
      int idx;
      if (arc.ilabel != 0) {  // emitting
        idx = e_idx++;
      } else {
        idx = ne_idx++;
      }
      h_arc_weights_[idx] = arc.weight.Value();
      h_arc_nextstates_[idx] = arc.nextstate;
      if (arc.nextstate == 0)
        printf("going to start: %i -> %i \n", i, arc.nextstate);
      // Converting ilabel here, to avoid reindexing when reading nnet3 output
      h_arc_id_ilabels_[idx] = arc.ilabel;
      int32 ilabel_pdf = trans_model.TransitionIdToPdf(arc.ilabel);
      h_arc_pdf_ilabels_[idx] = ilabel_pdf;
      h_arc_olabels_[idx] = arc.olabel;
    }
  }

  KALDI_DECODER_CUDA_API_CHECK_ERROR(
      cudaMemcpy(d_arc_weights_, &h_arc_weights_[0],
                 arc_count_ * sizeof(*d_arc_weights_), cudaMemcpyHostToDevice));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpy(
      d_arc_nextstates_, &h_arc_nextstates_[0],
      arc_count_ * sizeof(*d_arc_nextstates_), cudaMemcpyHostToDevice));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpy(
      d_arc_pdf_ilabels_, &h_arc_pdf_ilabels_[0],
      e_count_ * sizeof(*d_arc_pdf_ilabels_), cudaMemcpyHostToDevice));

  // Making sure the graph is ready
  cudaDeviceSynchronize();
  KALDI_DECODER_CUDA_CHECK_ERROR();
  nvtxRangePop();
}

void CudaFst::Finalize() {
  nvtxRangePushA("CudaFst destructor");
  CuDevice::Instantiate().Free(d_e_offsets_);
  CuDevice::Instantiate().Free(d_ne_offsets_);
  CuDevice::Instantiate().Free(d_final_);
  CuDevice::Instantiate().Free(d_arc_weights_);
  CuDevice::Instantiate().Free(d_arc_nextstates_);
  CuDevice::Instantiate().Free(d_arc_pdf_ilabels_);
  nvtxRangePop();
}
}  // end namespace CudaDecoder
}  // end namespace kaldi
