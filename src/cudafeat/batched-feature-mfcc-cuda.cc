// cudafeature/feature-mfcc-cuda.cu
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
#include "cudafeat/batched-feature-mfcc-cuda.h"
#include <nvToolsExt.h>
#include "cudamatrix/cu-rand.h"

namespace kaldi {

BatchedCudaMfcc::BatchedCudaMfcc(const MfccOptions &opts, const int batch_size,
                                 const int max_samples_chunks,
                                 const int max_states)
    : MfccComputer(opts),
      cu_lifter_coeffs_(lifter_coeffs_),
      cu_dct_matrix_(dct_matrix_),
      window_function_(opts.frame_opts),
      batch_size_(batch_size),
      max_states_(max_states),
      max_samples_chunks_(max_samples_chunks) {
  const MelBanks *mel_banks = GetMelBanks(1.0);
  const std::vector<std::pair<int32, Vector<BaseFloat>>> &bins =
      mel_banks->GetBins();  // feature banks
  int size = bins.size();
  bin_size_ = size;
  std::vector<int32> offsets(size), sizes(size);
  std::vector<float *> vecs(size);
  cu_vecs_ = new CuVector<float>[size];
  // pointer to GPU memory
  for (int i = 0; i < bins.size(); i++) {
    cu_vecs_[i].Resize(bins[i].second.Dim(), kUndefined);
    cu_vecs_[i].CopyFromVec(bins[i].second);
    vecs[i] = cu_vecs_[i].Data();
    sizes[i] = cu_vecs_[i].Dim();
    offsets[i] = bins[i].first;
  }
  offsets_ = static_cast<int32 *>(
      CuDevice::Instantiate().Malloc(size * sizeof(int32)));
  sizes_ = static_cast<int32 *>(
      CuDevice::Instantiate().Malloc(size * sizeof(int32)));
  vecs_ = static_cast<float **>(
      CuDevice::Instantiate().Malloc(size * sizeof(float *)));
  wave_offset = static_cast<int32 *>(
      CuDevice::Instantiate().Malloc(max_states_ * sizeof(int32)));

  CU_SAFE_CALL(cudaMemcpyAsync(vecs_, &vecs[0], size * sizeof(float *),
                               cudaMemcpyHostToDevice, cudaStreamPerThread));
  CU_SAFE_CALL(cudaMemcpyAsync(offsets_, &offsets[0], size * sizeof(int32),
                               cudaMemcpyHostToDevice, cudaStreamPerThread));
  CU_SAFE_CALL(cudaMemcpyAsync(sizes_, &sizes[0], size * sizeof(int32),
                               cudaMemcpyHostToDevice, cudaStreamPerThread));
  // memset wave_offset
  CU_SAFE_CALL(cudaMemsetAsync(wave_offset, 0, sizeof(int32) * max_states_,
                               cudaStreamPerThread));
  CU_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));

  frame_length_ = opts.frame_opts.WindowSize();
  padded_length_ = opts.frame_opts.PaddedWindowSize();
  fft_length_ = padded_length_ / 2;  // + 1;
  fft_size_ = 4800;

  // place holders to get strides for cufft.  these will be resized correctly
  // later.  The +2 for cufft/fftw requirements of an extra element at the end.
  // turning off stride because cufft seems buggy with a stride
  cu_windows_.Resize(fft_size_, padded_length_, kUndefined,
                     kStrideEqualNumCols);
  tmp_window_.Resize(fft_size_, padded_length_ + 2, kUndefined,
                     kStrideEqualNumCols);

  stride_ = cu_windows_.Stride();
  tmp_stride_ = tmp_window_.Stride();

  cufftPlanMany(&plan_, 1, &padded_length_, NULL, 1, stride_, NULL, 1,
                tmp_stride_ / 2, CUFFT_R2C, fft_size_);
  cufftSetStream(plan_, cudaStreamPerThread);

  // preallocation of state store
  // calculate how many samples we need to cache
  int frame_nums = NumFrames(max_samples_chunks_, opts.frame_opts, true);
  int frame_shift = opts.frame_opts.WindowShift();

  cache_num_samples = max_samples_chunks_ - frame_shift * frame_nums;
  total_dims = cache_num_samples + max_samples_chunks_;

  // states needed to cache
  chunk_states.Resize(max_states_, cache_num_samples, kUndefined,
                      kStrideEqualNumCols);
  // init
  chunk_states.SetZero();

  // copy pointers into device param;
  chunk_store.data = chunk_states.Data();
  chunk_store.stride = chunk_states.Stride();

  offset_store.data = wave_offset;
  offset_store.stride = 1;

  // copy into device struct
  device_params.chunk_channel = chunk_store;
  device_params.offset_channel = offset_store;
}

// reset channel to zero,
// only call kernel when needed
void BatchedCudaMfcc::ResetChannel(KernelParam *kernel_params,
                                   int batch_partial_size) {
  reset_states_mfcc(device_params, kernel_params, batch_partial_size,
                    cache_num_samples);
}

// save cached states
void BatchedCudaMfcc::SaveCachedStates(const FrameExtractionOptions &opts,
                                       KernelParam *kernel_params,
                                       int32 wave_dim, int batch_partial_size) {
  store_state_channel(batch_partial_size, device_params, kernel_params,
                      waveform_concat.Data(), waveform_concat.Stride(),
                      wave_dim, waveform_concat.NumCols(), opts.WindowShift(),
                      opts.WindowSize(), opts.snip_edges, cache_num_samples);
}

// basically this kernel will load states and concat into
// a new matrix
void BatchedCudaMfcc::LoadCachedStates(const CuMatrix<BaseFloat> &cu_wave,
                                       KernelParam *kernel_params,
                                       int batch_partial_size) {
  load_state_channel(batch_partial_size, device_params, kernel_params,
                     cu_wave.Data(), cu_wave.Stride(), waveform_concat.Data(),
                     waveform_concat.Stride(), cache_num_samples,
                     cu_wave.NumCols());
}

void BatchedCudaMfcc::ExtractWindows(int32_t num_frames, int64 sample_offset,
                                     CuMatrix<BaseFloat> &wave,
                                     const FrameExtractionOptions &opts,
                                     int batch_partial_size) {
  KALDI_ASSERT(sample_offset >= 0 && wave.NumRows() != 0 &&
               wave.NumCols() != 0);
  int32 frame_length = opts.WindowSize(),
        frame_length_padded = opts.PaddedWindowSize();

  extract_windows(num_frames, batch_partial_size, frame_length,
                  opts.WindowShift(), opts.WindowSize(), opts.snip_edges,
                  frame_length_padded, sample_offset, wave.Data(),
                  wave.Stride(), cu_windows_.Data(), cu_windows_.Stride());
}

// Kernel with block size 32 x 8 process a batches of windows
// Each block will process a batch size of 8 and a frames
// Applying Hamming Window, Pre-emphasis ,...
void BatchedCudaMfcc::ProcessWindows(int num_frames,
                                     const FrameExtractionOptions &opts,
                                     CuMatrix<BaseFloat> *log_energy_pre_window,
                                     int batch_partial_size) {
  if (num_frames == 0) return;

  int fft_num_frames = cu_windows_.NumRows();
  KALDI_ASSERT(fft_num_frames % fft_size_ == 0);

  process_window(num_frames, opts.dither, opts.remove_dc_offset,
                 opts.preemph_coeff, batch_partial_size, frame_length_,
                 NeedRawLogEnergy(), log_energy_pre_window->Data(),
                 window_function_.cu_window.Data(), tmp_window_.Data(),
                 tmp_window_.Stride(), cu_windows_.Data(),
                 cu_windows_.Stride());
}

// added channels to store frames (states) of MFCC
void BatchedCudaMfcc::ComputeFinalFeatures(
    int num_frames, int num_frames_batch, BaseFloat vtln_wrap,
    CuMatrix<BaseFloat> *cu_signal_log_energy,
    const FrameExtractionOptions &opts, CuMatrix<BaseFloat> *cu_features,
    int batch_partial_size) {
  Vector<float> tmp;
  assert(opts_.htk_compat == false);

  if (num_frames == 0) return;

  if (opts_.use_energy && !opts_.raw_energy) {
    dot_log_energy(num_frames, batch_partial_size, cu_windows_.NumCols(),
                   cu_windows_.Data(), cu_windows_.Stride(),
                   cu_signal_log_energy->Data());
  }

  // make sure a reallocation hasn't changed these
  KALDI_ASSERT(cu_windows_.Stride() == stride_);
  KALDI_ASSERT(tmp_window_.Stride() == tmp_stride_);

  // Perform FFTs in batches of fft_size.  This reduces memory requirements
  for (int idx = 0; idx < num_frames_batch; idx += fft_size_) {
    CUFFT_SAFE_CALL(cufftExecR2C(
        plan_, cu_windows_.Data() + cu_windows_.Stride() * idx,
        (cufftComplex *)(tmp_window_.Data() + tmp_window_.Stride() * idx)));
  }

  // Compute Power spectrum
  CuMatrix<BaseFloat> power_spectrum(tmp_window_.NumRows(),
                                     padded_length_ / 2 + 1, kUndefined);

  power_spectrum_compute(num_frames, batch_partial_size, padded_length_,
                         tmp_window_.Data(), tmp_window_.Stride(),
                         power_spectrum.Data(), power_spectrum.Stride());

  // mel banks
  int num_bins = bin_size_;
  cu_mel_energies_.Resize(num_frames_batch, num_bins, kUndefined);

  mel_features_bank_compute(num_bins, batch_partial_size, num_frames, offsets_,
                            sizes_, vecs_, power_spectrum.Data(),
                            power_spectrum.Stride(), cu_mel_energies_.Data(),
                            cu_mel_energies_.Stride());

  // dct transform
  // cu_features will have dimensions (num_frames_batch, feature_dims (bins))
  cu_features->AddMatMat(1.0, cu_mel_energies_, kNoTrans, cu_dct_matrix_,
                         kTrans, 0.0);

  // need to reset two last frames of first chunk to 0 for i-vector
  // need to reset last frames of last chunk to 0 for i-vector
  lifter_and_floor_energy_compute(
      num_frames, batch_partial_size, cu_features->Data(),
      cu_features->NumCols(), cu_features->Stride(),
      cu_signal_log_energy->Data(), opts_.cepstral_lifter, opts_.use_energy,
      opts_.energy_floor, cu_lifter_coeffs_.Data());
}

// Batch feature pipeline
// -cu_wave (batch_size x wave_dim)
// -cu_features (batch_size * num_frames x feature_dim)

// the reason starts needs to be part a kernel param
// is because threads in a block need to know the direction it will be writting
void BatchedCudaMfcc::ComputeBatchedFeatures(const CuMatrix<BaseFloat> &cu_wave,
                                             BaseFloat sample_freq,
                                             BaseFloat vtln_warp,
                                             CuMatrix<BaseFloat> *cu_features,
                                             KernelParam *kernel_params,
                                             int batch_partial_size) {
  nvtxRangePushA("CudaMfcc::ComputeBatchedFeatures");
  const FrameExtractionOptions &frame_opts = GetFrameOptions();
  int max_batch_size = cu_wave.NumRows();
  int num_frames =
      NumFrames(cu_wave.NumCols() + cache_num_samples, frame_opts, true);

  // compute fft frames by rounding up to a multiple of fft_size_
  int num_frames_batch = num_frames * max_batch_size;

  int fft_num_frames =
      num_frames_batch + (fft_size_ - num_frames_batch % fft_size_);
  int feature_dim = Dim();
  // bool use_raw_log_energy = NeedRawLogEnergy();

  CuMatrix<BaseFloat> raw_log_energies;
  // added batch_size dimensions, allocate data
  raw_log_energies.Resize(max_batch_size, num_frames, kUndefined);

  // allocate data

  cu_windows_.Resize(fft_num_frames, padded_length_, kUndefined,
                     kStrideEqualNumCols);
  //  cu_features->Resize(num_frames_batch, feature_dim, kUndefined,
  //                      kStrideEqualNumCols); TODO
  tmp_window_.Resize(fft_num_frames, padded_length_ + 2, kUndefined,
                     kStrideEqualNumCols);
  waveform_concat.Resize(batch_partial_size, total_dims, kUndefined,
                         kStrideEqualNumCols);

  if (frame_opts.dither != 0.0f) {
    // Calling cu-rand directly
    // CuRand class works on CuMatrixBase which must
    // assume that the matrix is part of a larger matrix
    // Doing this directly avoids unecessary memory copies

    CURAND_SAFE_CALL(
        curandGenerateNormal(GetCurandHandle(), tmp_window_.Data(),
                             tmp_window_.NumRows() * tmp_window_.Stride(),
                             0.0 /*mean*/, 1.0 /*stddev*/))
  }

  // new waveform data here (old states + new data)
  LoadCachedStates(cu_wave, kernel_params, batch_partial_size);

  // Extract Windows
  ExtractWindows(num_frames, 0, waveform_concat, frame_opts,
                 batch_partial_size);

  // Process Windows
  ProcessWindows(num_frames, frame_opts, &raw_log_energies, batch_partial_size);

  // Compute Features
  ComputeFinalFeatures(num_frames, num_frames_batch, 1.0, &raw_log_energies,
                       frame_opts, cu_features, batch_partial_size);

  // Store States
  SaveCachedStates(frame_opts, kernel_params, cu_wave.NumCols(),
                   batch_partial_size);

  nvtxRangePop();
}

BatchedCudaMfcc::~BatchedCudaMfcc() {
  delete[] cu_vecs_;
  CuDevice::Instantiate().Free(vecs_);
  CuDevice::Instantiate().Free(offsets_);
  CuDevice::Instantiate().Free(sizes_);
  CuDevice::Instantiate().Free(wave_offset);
  cufftDestroy(plan_);
}
}  // namespace kaldi
