// cudadecoder/batched-threaded-nnet3-cuda-online-pipeline.h
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Hugo Braun
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

#if HAVE_CUDA == 1

#ifndef KALDI_CUDA_DECODER_BATCHED_THREADED_CUDA_ONLINE_PIPELINE_H_
#define KALDI_CUDA_DECODER_BATCHED_THREADED_CUDA_ONLINE_PIPELINE_H_

#include <atomic>
#include <thread>

#include "base/kaldi-utils.h"
#include "cudadecoder/batched-static-nnet3.h"
#include "cudadecoder/cuda-decoder.h"
#include "cudadecoder/thread-pool.h"
#include "cudafeat/online-cuda-batched-feature-pipeline.h"
#include "feat/wave-reader.h"
#include "lat/determinize-lattice-pruned.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"
#include "online2/online-nnet2-feature-pipeline.h"

namespace kaldi {
namespace cuda_decoder {

struct BatchedThreadedNnet3CudaOnlinePipelineConfig {
  BatchedThreadedNnet3CudaOnlinePipelineConfig()
      : max_batch_size(400),
        num_channels(600),
        num_worker_threads(-1),
        determinize_lattice(true),
        num_decoder_copy_threads(2) {}
  void Register(OptionsItf *po) {
    po->Register("max-batch-size", &max_batch_size,
                 "The maximum batch size to be used by the decoder. "
                 "This is also the number of lanes in the CudaDecoder. "
                 "Larger = Faster and more GPU memory used.");
    po->Register("num-channels", &num_channels,
                 "The number of channels "
                 "allocated to the cuda decoder.  This should be larger "
                 "than max_batch_size.  Each channel consumes a small "
                 "amount of memory but also allows us to better overlap "
                 "computation.");
    po->Register("cuda-worker-threads", &num_worker_threads,
                 "(optional) The total number of CPU threads launched to "
                 "process CPU tasks. -1 = use std::hardware_concurrency()");
    po->Register("determinize-lattice", &determinize_lattice,
                 "Determinize the lattice before output.");
    po->Register("cuda-decoder-copy-threads", &num_decoder_copy_threads,
                 "Advanced - Number of worker threads used in the decoder for "
                 "the host to host copies.");
    feature_opts.Register(po);
    decoder_opts.Register(po);
    det_opts.Register(po);
    compute_opts.Register(po);
  }
  int max_batch_size;
  int num_channels;
  int num_worker_threads;
  bool determinize_lattice;
  int num_decoder_copy_threads;

  OnlineNnet2FeaturePipelineConfig feature_opts;
  CudaDecoderConfig decoder_opts;
  fst::DeterminizeLatticePhonePrunedOptions det_opts;
  nnet3::NnetSimpleComputationOptions compute_opts;
};

class BatchedThreadedNnet3CudaOnlinePipeline {
 public:
  using CorrelationID = uint64_t;
  BatchedThreadedNnet3CudaOnlinePipeline(
      const BatchedThreadedNnet3CudaOnlinePipelineConfig &config,
      const fst::Fst<fst::StdArc> &decode_fst,
      const nnet3::AmNnetSimple &am_nnet, const TransitionModel &trans_model)
      : config_(config),
        max_batch_size_(config.max_batch_size),
        trans_model_(&trans_model),
        am_nnet_(&am_nnet),
        word_syms_(NULL) {
    // TODO move that in config struct
    KALDI_ASSERT(config_.max_batch_size > 0);
    config_.compute_opts.CheckAndFixConfigs(am_nnet_->GetNnet().Modulus());
    int min_nchannels =
        config_.max_batch_size * 2;  // TODO handle available_channels_ empty
    config_.num_channels = std::max(config.num_channels, min_nchannels);

    int num_worker_threads = (config_.num_worker_threads > 0)
                                 ? config_.num_worker_threads
                                 : std::thread::hardware_concurrency();
    thread_pool_.reset(new ThreadPool(num_worker_threads));

    Initialize(decode_fst);
  }

  // Called when a new utterance will be decoded w/ correlation id corr_id
  // When this utterance will be done (when it will receive a chunk with
  // last_chunk=true)
  // Returns true if a channel was available
  bool InitCorrID(CorrelationID corr_id);

  // Set the callback function to call with the final lattice
  void SetLatticeCallback(
      CorrelationID corr_id,
      const std::function<void(CompactLattice &)> &callback);

  // Chunk of one utterance. We receive batches of those chunks through
  // DecodeBatch
  // Contains pointers to that chunk, the corresponding correlation ID, and
  // whether that chunk is the last one for that utterance
  struct UtteranceChunk {
    CorrelationID corr_id;
    SubVector<BaseFloat> wave_samples;
    bool last_chunk;  // sets to true if last chunk for that utterance
  };

  // Receive a batch of chunks. Will decode them, then return.
  // If it contains some last chunks for given utterances, it will call
  // FinalizeDecoding (building the final lattice, determinize it, etc.)
  // asynchronously. The callback for that utterance will then be called
  void DecodeBatch(const std::vector<CorrelationID> &corr_ids,
                   const std::vector<SubVector<BaseFloat>> &wave_samples,
                   const std::vector<bool> &is_last_chunk);
  // Version providing directly the features. Only runs nnet3 & decoder
  // Used when we want to provide the final ivectors (offline case)
  // channels can be provided if they are known (internal use)
  void DecodeBatch(const std::vector<CorrelationID> &corr_ids,
                   const std::vector<BaseFloat *> &d_features,
                   const int features_frame_stride,
                   const std::vector<int> &n_input_frames_valid,
                   const std::vector<BaseFloat *> &d_ivectors,
                   const std::vector<bool> &is_last_chunk,
                   std::vector<int> *channels = NULL);

  // Maximum number of samples per chunk
  int32 GetNSampsPerChunk() { return samples_per_chunk_; }
  int32 GetNInputFramesPerChunk() { return input_frames_per_chunk_; }
  float GetModelFrequency() { return model_frequency_; }
  int GetTotalNnet3RightContext() {
    return cuda_nnet3_->GetTotalNnet3RightContext();
  }
  // Maximum number of seconds per chunk
  BaseFloat GetSecondsPerChunk() { return seconds_per_chunk_; }

  // Used when debugging. Used to Print the text when a decoding is done
  void SetSymbolTable(fst::SymbolTable *word_syms) { word_syms_ = word_syms; }

  // Wait for all lattice callbacks to complete
  // Can be called after DecodeBatch
  void WaitForLatticeCallbacks();

 private:
  // Initiliaze this object
  void Initialize(const fst::Fst<fst::StdArc> &decode_fst);

  // Allocate and initialize data that will be used for computation
  void AllocateAndInitializeData(const fst::Fst<fst::StdArc> &decode_fst);

  // Reads what's needed from models, such as left and right context
  void ReadParametersFromModel();

  // Following functions are DecodeBatch's helpers

  // Filling  curr_batch_ichannels_
  void ListIChannelsInBatch(const std::vector<CorrelationID> &corr_ids,
                            std::vector<int> *channels);
  void CPUFeatureExtraction(
      const std::vector<int> &channels,
      const std::vector<SubVector<BaseFloat>> &wave_samples);

  // Compute features and ivectors for the chunk
  // curr_batch[element]
  // CPU function
  void ComputeOneFeature(int element);
  static void ComputeOneFeatureWrapper(void *obj, uint64_t element,
                                       uint64_t ignored) {
    static_cast<BatchedThreadedNnet3CudaOnlinePipeline *>(obj)
        ->ComputeOneFeature(element);
  }
  void RunNnet3(const std::vector<int> &channels,
                const std::vector<BaseFloat *> &d_features,
                const int feature_stride,
                const std::vector<int> &n_input_frames_valid,
                const std::vector<bool> &is_last_chunk,
                const std::vector<BaseFloat *> &d_ivectors);
  void RunDecoder(const std::vector<int> &channels);

  void BuildLatticesAndRunCallbacks(const std::vector<CorrelationID> &corr_ids,
                                    const std::vector<int> &channels,
                                    const std::vector<bool> &is_last_chunk);

  // If an utterance is done, we call FinalizeDecoding async on
  // the threadpool
  // it will call the utterance's callback when done
  void FinalizeDecoding(int32 ichannel, CorrelationID corr_id);
  // static wrapper for thread pool
  static void FinalizeDecodingWrapper(void *obj, uint64_t ichannel64,
                                      uint64_t corr_id) {
    int32 ichannel = static_cast<int32>(ichannel64);
    static_cast<BatchedThreadedNnet3CudaOnlinePipeline *>(obj)
        ->FinalizeDecoding(ichannel, corr_id);
  }
  // Data members

  BatchedThreadedNnet3CudaOnlinePipelineConfig config_;
  int32 max_batch_size_;  // extracted from config_
  // Models
  const TransitionModel *trans_model_;
  const nnet3::AmNnetSimple *am_nnet_;
  std::unique_ptr<OnlineNnet2FeaturePipelineInfo> feature_info_;
  // Decoder channels currently available, w/ mutex
  std::vector<int32> available_channels_;
  std::mutex available_channels_m_;

  // corr_id -> decoder channel map
  std::unordered_map<CorrelationID, int32> corr_id2channel_;

  const std::vector<int> *fe_threads_channels;
  const std::vector<SubVector<BaseFloat>> *fe_threads_wave_samples;

  // corr_id -> callback,  w/ mutex
  // the callback is called once the final lattice is ready
  std::unordered_map<int,
                     std::unique_ptr<std::function<void(CompactLattice &)>>>
      corrid2callbacks_;
  std::mutex corrid2callbacks_m_;

  // New channels in the current batch. We've just received
  // their first batch
  std::vector<int32> list_channels_first_chunk_;

  std::vector<int> n_samples_valid_, n_input_frames_valid_;

  std::vector<std::vector<std::pair<int, BaseFloat *>>>
      all_frames_log_posteriors_;

  // Channels done after current batch. We've just received
  // their last chunk
  std::vector<int> list_channels_last_chunk_;
  std::vector<CorrelationID> list_corr_id_last_chunk_;

  // Feature pipelines, associated to a decoder channel
  std::vector<std::unique_ptr<OnlineNnet2FeaturePipeline>> feature_pipelines_;

  // Number of frames already computed in channel (before
  // curr_batch_)
  std::vector<int32> channel_frame_offset_;

  // Parameters extracted from the models
  int input_frames_per_chunk_;
  int output_frames_per_chunk_;
  BaseFloat seconds_per_chunk_;
  BaseFloat samples_per_chunk_;
  BaseFloat model_frequency_;
  int32 ivector_dim_, input_dim_;

  // Buffers used during computation
  Matrix<BaseFloat> h_all_features_;
  Matrix<BaseFloat> h_all_waveform_;
  CuMatrix<BaseFloat> d_all_waveform_;
  CuMatrix<BaseFloat> d_all_features_;    // TODO add _
  CuMatrix<BaseFloat> d_all_cmvn_feats_;  // TODO remove
  Matrix<BaseFloat> h_all_ivectors_;
  CuMatrix<BaseFloat> d_all_ivectors_;
  CuMatrix<BaseFloat> d_all_log_posteriors_;

  std::atomic<int> n_compute_features_not_done_;
  std::atomic<int> n_lattice_callbacks_not_done_;

  std::unique_ptr<OnlineCudaBatchedFeaturePipeline> cuda_features_extractor_;
  std::unique_ptr<BatchedStaticNnet3> cuda_nnet3_;
  // HCLG graph : CudaFst object is a host object, but contains
  // data stored in
  // GPU memory
  std::shared_ptr<CudaFst> cuda_fst_;
  std::unique_ptr<CudaDecoder> cuda_decoder_;

  std::unique_ptr<ThreadPool> thread_pool_;

  // Used for debugging
  fst::SymbolTable *word_syms_;
  // Used when printing to stdout for debugging purposes
  std::mutex stdout_m_;
};

}  // end namespace cuda_decoder
}  // end namespace kaldi.

#endif  // KALDI_CUDA_DECODER_BATCHED_THREADED_CUDA_ONLINE_PIPELINE_H_
#endif  // HAVE_CUDA
