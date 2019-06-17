#ifndef CUDAFEAT_BATCHED_MFCC_KERNELS_H_
#define CUDAFEAT_BATCHED_MFCC_KERNELS_H_

#include "cudafeat/feature-window-cuda.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "feat/feature-mfcc.h"

#include "cudafeat/channel-defined.h"

namespace kaldi {

void lifter_and_floor_energy_compute(int num_frames, int batch_size, float *feats,
                                  int num_cols, int ldf, float *log_energy,
                                  float cepstral_lifter, bool use_energy, float energy_floor,
                                  float *lifter_coeffs);

  void power_spectrum_compute(int num_frames, int batch_size, int padded_length,
                            float *tmp_window, int ldt, float *power, int ldp);

  void mel_features_bank_compute(int num_bins, int batch_size, int num_frames,
                               int *offsets, int *sizes, float **vecs,
                               float *power, int ldp, float *energies,
                               int lde);

void process_window(int num_frames, float dither, bool remove_dc_offset,
                    float preemph_coeff, int batch_size, int frame_length, bool use_raw_energy,
                    float *log_energy, const float *windowing,
                    float *tmp_window, int ldt, float *window, int ldw);

void extract_windows(int num_frames, int batch_size, int frames_length,
                    int window_shift, int window_size, bool snip_edges,
                    int frame_length_padded, int sample_offset,
                    const float *wave, int ldw, float *window, int ldwd);

  void load_state_channel(int batch_size, DeviceMFCC device_params,
                        KernelParam *kernel_params, const float *wave, int ldw, 
                        float *waveform_concat, int lwc, int cache_col,
                        int chunk_cols);

void store_state_channel(int batch_size, DeviceMFCC device_params,
                        KernelParam *kernel_params, float *waveform, int ldw,
                        int num_samples, int num_samples_concat, int window_shift,
                        int window_size, bool snip_edges, int cache_num_samples);

  void dot_log_energy(int num_frames, int batch_size, int num_cols, float *windows,
                    int ldw, float *energy);

  void reset_states_mfcc(DeviceMFCC device_params, 
                      KernelParam *kernel_params, int batch_size,
                      int wave_dim);


}
#endif
