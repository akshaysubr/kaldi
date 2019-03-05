// cudadecoder/cuda-fst.h
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

#ifndef KALDI_DECODER_CUDA_FST_H_
#define KALDI_DECODER_CUDA_FST_H_
#include "cudadecoder/cuda-decodable-itf.h"
#include "cudamatrix/cu-device.h"
#include "lat/kaldi-lattice.h"
#include "nnet3/decodable-online-looped.h"
#include "util/stl-utils.h"

namespace kaldi {
namespace CudaDecoder {
typedef fst::StdArc StdArc;
typedef StdArc::Weight StdWeight;
typedef StdArc::Label Label;
typedef StdArc::StateId StateId;
// FST in device memory
// This class is based on the Compressed Sparse Row (CSR) Matrix format.
// https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)
// Where states = rows and arcs = columns.
// Emitting arcs and non-emitting arcs are stored as seperate matrices for
// efficiency
class CudaFst {
 public:
  CudaFst(){};
  // Creates a CSR representation of the FST,
  // then copies it to the GPU
  void Initialize(const fst::Fst<StdArc> &fst,
                  const TransitionModel &trans_model);
  void Finalize();

  inline uint32_t NumStates() const { return num_states_; }
  inline StateId Start() const { return start_; }

 private:
  friend class CudaDecoder;

  // counts arcs and computes offsets of the fst passed in
  void ComputeOffsets(const fst::Fst<StdArc> &fst);

  // allocates memory to store FST
  void AllocateData(const fst::Fst<StdArch> &fst);

  // copies fst into the pre-allocated datastructures
  void CopyData(const fst::Fst<StdArc> &fst);
  // Total number of states
  unsigned int num_states_;

  // Starting state of the FST
  // Computation should start from state start_
  StateId start_;

  // Number of emitting, non-emitting, and total number of arcs
  unsigned int e_count_, ne_count_, arc_count_;

  // This data structure is similar to a CSR matrix format
  // with 2 offsets matrices (one emitting one non-emitting).

  // Offset arrays are num_states_+1 in size (last state needs
  // its +1 arc_offset)
  // Arc values for state i are stored in the range of [offset[i],offset[i+1][

  unsigned int *d_e_offsets_;  // Emitting offset arrays
  std::vector<unsigned int> h_e_offsets_;
  unsigned int *d_ne_offsets_;  // Non-emitting offset arrays
  std::vector<unsigned int> h_ne_offsets_;

  // These are the values for each arc.
  // Arcs belonging to state i are found in the range of [offsets[i],
  // offsets[i+1][
  // Use e_offsets or ne_offsets depending on what you need
  // (emitting/nonemitting)
  // The ilabels arrays are of size e_count_, not arc_count_

  std::vector<float> h_arc_weights_;
  float *d_arc_weights_;                   // TODO define CostType here
  std::vector<StateId> h_arc_nextstates_;  // TODO remove "s"
  StateId *d_arc_nextstates_;
  std::vector<int32> h_arc_id_ilabels_;
  int32 *d_arc_pdf_ilabels_;
  std::vector<int32> h_arc_olabels_;

  // Final costs
  // final cost of state i is h_final_[i]
  std::vector<float> h_final_;
  float *d_final_;
};
}  // end namespace CudaDecoder
}  // end namespace kaldi
#endif
