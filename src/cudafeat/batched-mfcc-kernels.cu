#include "cudafeat/batched-mfcc-kernels.h"
#include <cub/cub.cuh>


#define FPTHREADS 4


namespace kaldi {

//basically this two device functions are 
//just copy from Kaldi

__device__ inline int32 FirstSampleOfFrame(int32 frame, int32 frame_shift,
                                           int32 window_size, bool snip_edges) {
  if (snip_edges) {
    return frame * frame_shift;
  } else {
    int32 midpoint_of_frame = frame_shift * frame + frame_shift / 2,
          beginning_of_frame = midpoint_of_frame - window_size / 2;
    return beginning_of_frame;
  }
}

__device__ inline int32 NumFramesDevice(int64 num_samples, bool flush, 
                        int32 frame_shift, int32 frame_length,
                        bool snip_edges) {
  if (snip_edges) {
    if (num_samples < frame_length)
      return 0;
    else
      return (1 + ((num_samples - frame_length) / frame_shift));
  } else {
    int32 num_frames = (num_samples + (frame_shift / 2)) / frame_shift;
    if (flush)
      return num_frames;
    
    int64 end_sample_of_last_frame = FirstSampleOfFrame(num_frames - 1, frame_shift,
        frame_length, snip_edges) + frame_length;

    while (num_frames > 0 && end_sample_of_last_frame > num_samples) {
      num_frames--;
      end_sample_of_last_frame -= frame_shift;
    }
    return num_frames;
  }
}


// Each thread block processes a unique frame
// threads in the same threadblock collaborate to
// compute the frame together.

// Batched lifter_and_floor_energy
// - num_cols: feature_dims
// - log_energy: raw_log_energy (batch_size x num_frames)
// - features: cu_features (num_frames_batch x feature_dim)
// - ldf: stride into features

__global__ void batched_apply_lifter_and_floor_energy(
    int num_frames, int num_cols, float cepstral_lifter, bool use_energy,
    float energy_floor, float *log_energy, float *lifter_coeffs,
    float *features, int32_t ldf, int32_t batch_size) {

  int thread_id = threadIdx.x;
  int frame = blockIdx.x;
  
  int bidx = blockDim.y * blockIdx.y + threadIdx.y; //samples id inside a batch

  if(bidx >= batch_size) return;

  float *feats = features + ldf * (gridDim.x * bidx + frame);

  // apply lifter coefficients
  if (cepstral_lifter != 0.0f) {
   for (int c = thread_id; c < num_cols; c += blockDim.x) {
     float lift = lifter_coeffs[c];
     float f = feats[c];
     feats[c] = f * lift;
   }
  }

   // Thread 0 for each frame will apply energy
   if (use_energy && thread_id == 0) {
     float *log_e = log_energy + gridDim.x * bidx;
     float energy = log_e[frame];
     float log_energy_floor = log(energy_floor);

     if (energy_floor > 0.0f && energy < log_energy_floor) {
        energy = log_energy_floor;
     }
     feats[0] = energy;
   }
}


// Each threadblock computes a different row of the matrix.
// Threads in the same block compute the row collaboratively.
// This kernel must be called out of place (A_in!=A_out).

// -  A_in: tmp_window (fft_num_frames * (padded_length + 2))
// -  ldi: stride of tmp_window
// -  A_out: power_spectrum matrix (fft_num_frames * padded_length / 2 + 1)
// -  ldo: stride of power spectrum

__global__ void batched_power_spectrum_kernel(int row_length, float *A_in, int32_t ldi,
                                      float *A_out, int32_t ldo, int32_t batch_size) {
    int thread_id = threadIdx.x;
    int frame = blockIdx.x;
  
 
    int bidx = blockDim.y * blockIdx.y + threadIdx.y; //samples id inside the batch

    if(bidx >= batch_size) return;

    float *Ar = A_in + ldi * (gridDim.x * bidx + frame);
    float *Aw = A_out + ldo * (gridDim.x * bidx + frame);

    int half_length = row_length / 2;
    for (int idx = thread_id; idx < half_length; idx += blockDim.x) {
      // ignore special case
      if (idx == 0) continue;

      float2 val = reinterpret_cast<float2 *>(Ar)[idx];
      float ret = val.x * val.x + val.y * val.y;
      Aw[idx] = ret;
    }

    // handle special case
    if (threadIdx.x == 0) {
      float real = Ar[0];
      // cufft puts this at the end, this is different than kaldi does with its
      // own
      // internal implementation
      float im = Ar[row_length];

      Aw[0] = real * real;
      Aw[half_length] = im * im;
   }
}

// Expects to be called with 32x8 sized thread block.
// Batched mel_computation
// Each thread blocks will compute one bins for 8 batches
// - offsets, sizes, vecs: (vecs: filter_bank)
// - feats: power_spectrum (fft_num_frames * ( padded_length / 2 + 1 ))
// - mels: mel outputs: cu_mel_energies (num_frames_batch * num_bins)
// - ldf, ldm: leading dimension
__global__ void batched_mel_banks_compute_kernel(int32_t num_frames, float energy_floor,
                                         int32 *offsets, int32 *sizes,
                                         float **vecs, const float *feats,
                                         int32_t ldf, float *mels,
                                         int32_t ldm, int32_t batch_size) {
  typedef cub::WarpReduce<float> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage[8];
  //we calculate a batch, so need a loop over frames
  int thread_id = threadIdx.x;
  int bin = blockIdx.x;

  int frame_group = blockIdx.z;

  int bidx = blockDim.y * blockIdx.y + threadIdx.y;  //get id of samples in batch;

  if(bidx >= batch_size) return;

  int wid = threadIdx.y;

  //offset into big matrices
  int offset = offsets[bin];
  int size = sizes[bin];

  const float *v = vecs[bin];

  for(int frame = 0 ; frame < FPTHREADS ; frame++) {
 
      //offset into power spectrum
      int frame_offset = frame_group * FPTHREADS + frame;
      if(frame_offset >= num_frames) return;
      const float *w = feats + ldf * (num_frames * bidx + frame_offset)
                               + offset;

      //local accumulation
      float sum = 0;
      for(int tidx = thread_id ; tidx < size ; tidx += 32) {
        sum += v[tidx] * w[tidx];
      }

      //warpReduce
      sum = WarpReduce(temp_storage[wid]).Sum(sum);
      if(thread_id == 0) {
        //avoid log_of_zero
        if(sum < energy_floor) sum = energy_floor;
        float val = logf(sum);
        int index = ldm * (num_frames * bidx + frame_offset) + bin;
        //offset into mels matrix
        mels[index] = val;
      }

      //sync threads to reuse temp storage
      __syncthreads();
   }
}



// Batched process_window:
// - log_energy_pre_window (batch_size x num_frames)
// - tmp_windows (fft_num_frames x (padded_length + 2))
// - windows (fft_num_frames x padded_length)
// - ldt, ldw: num_cols
// - windowing: hamming windows features (frame_length)

// DONE


//everything is warp level
__global__ void batched_process_window_kernel(
    int frame_length, float dither, float energy_floor, bool remove_dc_offset,
    float preemph_coeff, bool need_raw_log_energy, float *log_energy_pre_window,
    const float *windowing, float *tmp_windows, int32_t ldt, float *windows,
    int32_t ldw, int32_t batch_size) {
  // Specialize BlockReduce for type float
  typedef cub::WarpReduce<float> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage[8];

  int thread_id = threadIdx.x; //thread id in warp
  int frame = blockIdx.x; 
  int wid = threadIdx.y;

  int bidx = blockDim.y * blockIdx.y + threadIdx.y;   //get id of samples in batch

  
  if(bidx >= batch_size) return;
  
  //offset into big matrices
  float *tmp_window = tmp_windows + ldt * (gridDim.x * bidx + frame);
  float *window = windows + ldw *(gridDim.x * bidx + frame);

  __shared__ float ssum[8]; //8 warps

  float sum = 0;
  float wdot = 0;

  for (int idx = thread_id; idx < frame_length; idx += blockDim.x) {
    // tmp_window contains optional dither.  Apply that on read.
    float wval = window[idx];
    if (dither != 0.0f) {
      wval += tmp_window[idx] * dither;
    }
    // compute local sum for removing dc offset
    sum += wval;
    // compute dot product for log energy
    wdot += wval * wval;

    float windowing_mul = 1;
    if (remove_dc_offset == false && preemph_coeff == 0.0f) {
      // we are done here so set windowing multiplication on write.
      windowing_mul = windowing[idx];
    }

    // write dithered output
    window[idx] = wval * windowing_mul;
  }
  __syncthreads();


  if (remove_dc_offset) {
   // we will recompute this below
    wdot = 0.0f;
    // use cub to reduce
    sum = WarpReduce(temp_storage[wid]).Sum(sum); //sum is local to each thread


    // broadcast sum to entire warp
    if (thread_id == 0) ssum[wid] = sum;
    __syncthreads();

    sum = -ssum[wid] / frame_length;
    for (int idx = thread_id; idx < frame_length; idx += blockDim.x) {
      float windowing_mul = 1;
      float *out = window;
      if (preemph_coeff == 0.0f) {
        // we are done here so apply windowing
        windowing_mul = windowing[idx];
      } else {
        // write to temp window as we will copy back into window
        // when doing pre-emphasis
        out = tmp_window;
      }
      // updated window value
      float wval = window[idx] + sum;

      // compute new dot product with dc offset removed
      wdot += wval * wval;

      assert(windowing_mul == 1);
      // write output
      out[idx] = wval * windowing_mul;
    }
  }
  __syncthreads();

  // if pointer is not NULL we will set energy to either
  // the computed energy or 0 depending on need_raw_log_energy
  if (log_energy_pre_window != NULL) {
    float energy = 0.0f;

    // compute offset to log_energy matrix
    float *log_energy_window = log_energy_pre_window + bidx * gridDim.x;

    if (need_raw_log_energy) {
      // must sync to use retemp_storage
      if (remove_dc_offset) __syncthreads();
      // use cub to reduce
      wdot = WarpReduce(temp_storage[wid]).Sum(wdot);

      energy = max(wdot, energy_floor);
    }

    if (thread_id == 0) {
      log_energy_window[frame] = log(energy);
    }
  }

  // TODO this could be more efficient using shared memory instead of
  // tmp_window.
  if (preemph_coeff != 0.0f) {
    // wait for tmp_window to be computed
    __threadfence();
    __syncthreads();
    // starting thread idx at 0 to keep writes aligned.
    // unaligned reads are less painful then unaligned writes
    for (int idx = thread_id; idx < frame_length; idx += blockDim.x) {
      float wval = tmp_window[idx];
      float prev_window = wval;
      if (idx > 0) {
        prev_window = tmp_window[idx - 1];
      }
      // use __fmul_rn to match CPU
      // window[idx] = (wval - preemph_coeff*prev_window) * windowing[idx];
      window[idx] =
        (wval - __fmul_rn(preemph_coeff, prev_window)) * windowing[idx];
    }
  }
}

// batched window extract: (across utterances)
// - wave: (batch_size x wave_dim) (100 * 3200)
// - windows:(batch_size * num_frames x frames_length) 
// - wlda: stride into windows matrix
// - wave_dim: num of samples for wave

///DONE: should be correct

__global__ void batched_extract_window_kernel(
    int32 frame_shift, int32 frame_length, int32 frame_length_padded,
    int32 window_size, bool snip_edges, int32_t sample_offset,
    const BaseFloat *__restrict__ wave, int32 wave_dim,
    BaseFloat *__restrict__ windows, int32_t wlda, int32_t batch_size) {
 
  int frame = blockIdx.x; 
  int tidx = threadIdx.x;  

  int bidx = blockIdx.y * blockDim.y + threadIdx.y;   //corresponds to samples id in the batch

  if(bidx >= batch_size) return;

  int bwave_dim = bidx * wave_dim; // wave dimensions in the batched matrix

  int32 start_sample_in_block = 
    FirstSampleOfFrame(frame, frame_shift, window_size, snip_edges);

  int32 start_sample = start_sample_in_block + bwave_dim;

  // wave_start and wave_end are start and end indexes into 'wave', for the
  // piece of wave that we're trying to extract.
  int32 wave_start = int32(start_sample - sample_offset),
      wave_end = wave_start + frame_length;

  //offset into the windows matrix
  BaseFloat *window = windows +  wlda * (gridDim.x * bidx + frame);
  if (wave_start >= bwave_dim && wave_end <= bwave_dim + wave_dim) {
    // the normal case-- no edge effects to consider.
     for (int i = tidx; i < frame_length; i += blockDim.x) {
       window[i] = wave[wave_start + i];
     }
   } else {
     // Deal with any end effects by reflection, if needed.  This code will only
     // be reached for about two frames per utterance, so we don't concern
     // ourselves excessively with efficiency.
     for (int s = tidx; s < frame_length; s += blockDim.x) {
       int32 s_in_wave = s + wave_start;
       while (s_in_wave < bwave_dim || s_in_wave >= (bwave_dim + wave_dim)) {
         // reflect around the beginning or end of the wave.
         // e.g. -1 -> 0, -2 -> 1.
         // dim -> dim - 1, dim + 1 -> dim - 2.
         // the code supports repeated reflections, although this
         // would only be needed in pathological cases.
         if (s_in_wave < bwave_dim)
           s_in_wave = -(s_in_wave - bwave_dim) - 1;
         else
           s_in_wave = 2 * (bwave_dim + wave_dim) - 1 - s_in_wave;
       }
       window[s] = wave[s_in_wave];
     } 
   }

   if (frame_length_padded > frame_length) {
     for (int i = frame_length + tidx; i < frame_length_padded;
         i += blockDim.x) {
       window[i] = 0.0f;
     }
   }
 }



// For each frame
//   compute logf(dot(signal_frame, signal_frame))
// - signal_frame: cu_windows_  (fft_num_frames * padded_length)
// - signal_log_energy: raw_log_energy (log energy)


// DONE

__global__ void batched_dot_log_kernel(int32_t num_frames, int32_t frame_length,
                               float *signal_frame, int32_t lds,
                               float *signal_log_energy, int32_t batch_size) {
  // Specialize WarpReduce for type float
  // Allocate WarpReduce shared memory for 8 warps
  typedef cub::WarpReduce<float> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage[8];

  int32_t frame = blockIdx.x;
  int32_t tid = threadIdx.x;
  int32_t wid = threadIdx.y;

  int32_t bidx = blockDim.y * blockIdx.y + threadIdx.y;  // samples index inside the batch

  if(bidx >= batch_size) return;

  float *in = signal_frame + lds * (gridDim.x * bidx + frame);
  float sum = 0;

  // preform local dot product
  for (int32_t i = tid; i < frame_length; i += blockDim.x) {
      float val = in[i];
      sum += val * val;
   }

   // reduce using cub
   sum = WarpReduce(temp_storage[wid]).Sum(sum);

   if (threadIdx.x == 0) {
      signal_log_energy[frame] = logf(sum);
   }
}


//this kernel will load the states and build batch
//basically concat old states with new batch
//num_col: number of samples of past states
//num_cols: number of samples of current chunk
__global__ void batched_load_states_and_build_batch_kernel(DeviceMFCC device_params, 
                            KernelParam *kernel_params, const float *waveform, int ldw, 
                            float *waveform_cached, int lwc, int batch_size, 
                            int num_col, int num_cols) {
  int bidx = blockIdx.x; //sample id in batch

  if(bidx >= batch_size) return;
  //channel id and whether start
  //the first chunk will be handled differently
  Channels channels = kernel_params[bidx].compute_channel;
  int32 starts = kernel_params[bidx].start_utt;

  //offset
  float *state = device_params.chunk_channel.lane(channels);
  const float *wave = waveform + ldw * bidx;
  float *wave_cached = waveform_cached + lwc * bidx;

  if(starts) {
      //load only the part belong to new data
    for(int col = threadIdx.x ; col < num_cols ; col += blockDim.x) {
      //never goes off
      if(col >= num_cols) return;
      //store
      wave_cached[col] = wave[col];
    }
  }
  else {
    //load past states as well as new states
    for(int col = threadIdx.x ; col < num_col ; col += blockDim.x) {
      //never goes off
      if(col < num_col) {
        //store
        wave_cached[col] = state[col];
      }
    }
    //finish first phase
    for(int col = threadIdx.x ; col < num_cols ; col += blockDim.x) {
      //never goes off
      if(col >= num_cols) return;
      //store with offset numcol
      wave_cached[col + num_col] = wave[col];
    }
  }
}



//calculates states in mfcc that we need to store
//for next call
__global__ void batched_store_states_and_calculate_offset_kernel(DeviceMFCC device_params,
                                KernelParam *kernel_params, float *waveform, int ldw, int num_samples,
                                int32 num_samples_concat, int32 frame_shift, int32 frame_length,
                                bool snip_edges, int num_col) {
  int bidx = blockIdx.x;
  //channel id
  Channels channels = kernel_params[bidx].compute_channel;
  int32 starts = kernel_params[bidx].start_utt;

  int32 num_samp = (starts == 0) ? num_samples_concat : num_samples;

  //offset
  float *wave = waveform + ldw * bidx;
  float *state = device_params.chunk_channel.lane(channels);

  //load states offset and num_frames_old for this channel
  int32 *wave_offset = device_params.offset_channel.lane(channels);

  //number of samples seen so far;
  int64 num_samples_total = wave_offset[0] + num_samp;
  int32 num_frames_new = NumFramesDevice(num_samples_total, true, frame_shift, 
                                frame_length, snip_edges);
  //calculate next frame sample
  int64 first_sample_of_next_frame = FirstSampleOfFrame(num_frames_new, frame_shift, 
                                      frame_length, snip_edges);

  int32 samples_to_discard = first_sample_of_next_frame - wave_offset[0];

  if (samples_to_discard > 0) {
    // discard the leftmost part of the waveform that we no longer need.
    int new_num_samples = num_samp - samples_to_discard;

    if (new_num_samples <= 0) {
      //basically no caching of state
      if(threadIdx.x == 0) {
        //offset 
        wave_offset[0] += num_samp;
      }
      //reset everything in the state
      for(int col = threadIdx.x ; col < num_col ; col += blockDim.x) {
        //never goes off
        if(col >= num_col) return;
        //store
        state[col] = 0;
      }
    } 
    else {
      //we have to cache state and also copy frames
      //samples_to_discard are also where we offset
      if(threadIdx.x == 0) {
        //offset increment
        wave_offset[0] += samples_to_discard;
      }
      //each thread store
      for(int col = threadIdx.x ; col < num_col ; col += blockDim.x) {
        //never goes off
        if(col >= num_col) return;
        //store
        state[col] = wave[col + samples_to_discard];
      }
    }
  }
}

//This kernel will reset state of mfcc
__global__ void reset_state_batched_online_mfcc(DeviceMFCC device_params, 
                              KernelParam *kernel_params, int32 wave_dim) {
  //reset state
  int bidx = blockIdx.x;

  //channels to compute and end
  Channels ichannel = kernel_params[bidx].compute_channel;
  int32 end_chunk = kernel_params[bidx].end_utt;

  //offset pointers
  float *wave = device_params.chunk_channel.lane(ichannel);
  int32 *wave_offset = device_params.offset_channel.lane(ichannel);

  if(end_chunk) {
    //only reset when end of utt
    for(int tid = threadIdx.x ; tid < wave_dim ; tid += blockDim.x) {
      //reset wave cache
      if(tid < wave_dim) {
        wave[tid] = 0.0f;
      }
    }
    //reset wave_offset
    if(threadIdx.x == 0) {
      //only thread 0 gets here
      wave_offset[0] = 0;
    }
  }
}


void reset_states_mfcc(DeviceMFCC device_params, 
                      KernelParam *kernel_params, int batch_size,
                      int wave_dim) {
  //reset
  reset_state_batched_online_mfcc<<<batch_size, 512>>>(
    device_params, kernel_params, wave_dim);
  CU_SAFE_CALL(cudaGetLastError());
}


void lifter_and_floor_energy_compute(int num_frames, int batch_size, float *feats,
                                  int num_cols, int ldf, float *log_energy,
                                  float cepstral_lifter, bool use_energy, float energy_floor,
                                  float *lifter_coeffs) {
  //apply_lifter_kernel
  dim3 energy_threads(32 , 8);
  dim3 energy_blocks(num_frames, (batch_size + energy_threads.y - 1) / energy_threads.y);

  batched_apply_lifter_and_floor_energy<<<energy_blocks, energy_threads>>>( 
      num_frames, num_cols, cepstral_lifter,
      use_energy, energy_floor, log_energy,
      lifter_coeffs, feats, ldf, batch_size);
  CU_SAFE_CALL(cudaGetLastError());
}

void power_spectrum_compute(int num_frames, int batch_size, int padded_length,
                            float *tmp_window, int ldt, float *power, int ldp) {

  //power spectrum compute
  dim3 power_threads(32 , 8);
  dim3 power_blocks(num_frames, (batch_size + power_threads.y - 1) / power_threads.y);

  batched_power_spectrum_kernel<<<power_blocks, power_threads>>>(
      padded_length, tmp_window, ldt,
      power, ldp, batch_size);
  CU_SAFE_CALL(cudaGetLastError());
}

void mel_features_bank_compute(int num_bins, int batch_size, int num_frames,
                               int *offsets, int *sizes, float **vecs,
                               float *power, int ldp, float *energies,
                               int lde) {

  //compute mel filter bank;
  dim3 mel_threads(32 , 8, 1);
  dim3 mel_blocks(num_bins, (batch_size + mel_threads.y - 1) / mel_threads.y, (num_frames + FPTHREADS - 1) / FPTHREADS);
  batched_mel_banks_compute_kernel<<<mel_blocks, mel_threads>>>(
      num_frames, std::numeric_limits<float>::epsilon(), offsets, sizes,
      vecs, power, ldp, energies, lde, batch_size);
  CU_SAFE_CALL(cudaGetLastError());
}

void process_window(int num_frames, float dither, bool remove_dc_offset,
                    float preemph_coeff, int batch_size, int frame_length, bool use_raw_energy,
                    float *log_energy, const float *windowing,
                    float *tmp_window, int ldt, float *window, int ldw) {

  //process window with hamming
  dim3 process_threads(32 , 8);     
  dim3 process_blocks(num_frames, (batch_size + process_threads.y - 1) / process_threads.y);

  batched_process_window_kernel<<<process_blocks, process_threads>>>(
      frame_length, dither, std::numeric_limits<float>::epsilon(),
      remove_dc_offset, preemph_coeff, use_raw_energy,
      log_energy, windowing,
      tmp_window, ldt, window, ldw, batch_size);

  CU_SAFE_CALL(cudaGetLastError())
}

void extract_windows(int num_frames, int batch_size, int frames_length,
                    int window_shift, int window_size, bool snip_edges,
                    int frame_length_padded, int sample_offset,
                    const float *wave, int ldw, float *window, int ldwd) {

  //extract window, framing
  dim3 extract_threads(32, 8);
  dim3 extract_blocks(num_frames, (batch_size + extract_threads.y - 1) / extract_threads.y);


  batched_extract_window_kernel<<<extract_blocks ,extract_threads>>>(
      window_shift, frames_length, frame_length_padded, window_size,
      snip_edges, sample_offset, wave, ldw,
      window, ldwd, batch_size);
  CU_SAFE_CALL(cudaGetLastError());
}

void load_state_channel(int batch_size, DeviceMFCC device_params,
                        KernelParam *kernel_params, const float *wave, int ldw, 
                        float *waveform_concat, int lwc, int cache_col,
                        int chunk_cols) {

  //load states of online mfcc
  batched_load_states_and_build_batch_kernel<<<batch_size, 512>>>(
    device_params, kernel_params, wave,
    ldw, waveform_concat, lwc, batch_size, cache_col, 
    chunk_cols);
  CU_SAFE_CALL(cudaGetLastError());
}

void store_state_channel(int batch_size, DeviceMFCC device_params,
                        KernelParam *kernel_params, float *waveform, int ldw,
                        int num_samples, int num_samples_concat, int window_shift,
                        int window_size, bool snip_edges, int cache_num_samples) {

  //store the states of online mfcc
  batched_store_states_and_calculate_offset_kernel<<<batch_size, 512>>>(
    device_params, kernel_params, waveform,
    ldw, num_samples, num_samples_concat, window_shift, 
    window_size, snip_edges,
    cache_num_samples);
  CU_SAFE_CALL(cudaGetLastError());
}

void dot_log_energy(int num_frames, int batch_size, int num_cols, float *windows,
                    int ldw, float *energy) {

  //calculate log energy
  dim3 log_threads(32 , 8);
  dim3 log_blocks(num_frames, (batch_size + log_threads.y - 1) / log_threads.y);

  batched_dot_log_kernel<<<log_blocks, log_threads>>>(
        num_frames, num_cols, windows, ldw, energy, batch_size);
  CU_SAFE_CALL(cudaGetLastError())
}
}