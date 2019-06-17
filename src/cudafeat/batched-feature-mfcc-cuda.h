// cudafeat/feature-mfcc-cuda.h
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Justin Luitjens
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

#ifndef KALDI_CUDAFEAT_BATCHED_FEATURE_MFCC_CUDA_H_
#define KALDI_CUDAFEAT_BATCHED_FEATURE_MFCC_CUDA_H_

#if HAVE_CUDA == 1
#include <cufft.h>
#endif

#include "cudafeat/batched-mfcc-kernels.h"


namespace kaldi {
// This class implements MFCC computation in CUDA.
// It takes input from device memory and outputs to
// device memory.  It also does no synchronization.

class BatchedCudaMfcc : public MfccComputer {
 public:
    void ComputeBatchedFeatures(const CuMatrix<BaseFloat> &cu_wave,
                           BaseFloat sample_freq, BaseFloat vtln_warp,
                           CuMatrix<BaseFloat> *cu_features,
                           KernelParam *kernel_params, int
                           batch_partial_size);
  //preallocation
  BatchedCudaMfcc(const MfccOptions &opts, const int batch_size, 
                  const int max_samples_chunks_, 
                  const int max_states_);
  ~BatchedCudaMfcc();

  CuMatrix<float> cu_windows_; //batching
  CuMatrix<float> tmp_window_;

  //function for online mfcc
  void ResetChannel(KernelParam *kernel_params,
                      int batch_partial_size);
  

 private:

  void ExtractWindows(int32 num_frames, int64 sample_offset,
                      CuMatrix<BaseFloat> &wave,
                      const FrameExtractionOptions &opts,
                      int batch_partial_size);

  void ProcessWindows(int num_frames, const FrameExtractionOptions &opts,
                      CuMatrix<BaseFloat> *log_energy_pre_window,
                      int batch_partial_size);

  void ComputeFinalFeatures(int num_frames, int num_frames_batch,
                            BaseFloat vtln_wrap, CuMatrix<BaseFloat> *cu_signal_log_energy,
                            const FrameExtractionOptions &opts,
                            CuMatrix<BaseFloat> *cu_features,
                            int batch_partial_size);

  void SaveCachedStates(const FrameExtractionOptions &opts, 
                        KernelParam *kernel_params, int32 wave_dim,
                        int batch_partial_size);
  void LoadCachedStates(const CuMatrix<BaseFloat> &cu_wave,
                        KernelParam *kernel_params,
                        int batch_partial_size);

  CuMatrix<float> cu_mel_energies_;
  CuVector<float> cu_lifter_coeffs_;
  CuMatrix<float> cu_dct_matrix_;


  //concat samples cached
  CuMatrix<float> waveform_concat;


  int frame_length_, padded_length_, fft_length_, fft_size_;
  cufftHandle plan_;
  CudaFeatureWindowFunction window_function_;

  int bin_size_;
  int32 *offsets_, *sizes_;
  CuVector<float> *cu_vecs_;
  float **vecs_;

  int batch_size_;

  //max_states stored
  int max_states_;

  //max_samples per chunk
  int max_samples_chunks_;

  // for sanity checking cufft
  int32_t stride_, tmp_stride_;

  int cache_num_samples;
  int total_dims;

  int invalid_start_frames;

  //store states of mfcc (some last frames);
  CuMatrix<BaseFloat> chunk_states;

  //store offset, device pointer to int arr
  int32 *wave_offset;


  //lane will hold pointers
  StateCaching<float> chunk_store;
  StateCaching<int32> offset_store;

  //device param
  DeviceMFCC device_params;

  //Kernel Param
  KernelParam *kernel_params;

};


}

#endif
