// cudadecoder/batched-threaded-nnet3-cuda-pipeline.h
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

#if HAVE_CUDA == 1

#ifndef KALDI_CUDA_DECODER_BATCHED_THREADED_CUDA_DECODER_H_
#define KALDI_CUDA_DECODER_BATCHED_THREADED_CUDA_DECODER_H_

#include <atomic>
#include <thread>

#include "cudadecoder/batched-threaded-nnet3-cuda-online-pipeline.h"
#include "cudadecoder/cuda-decoder.h"
#include "cudafeat/online-cuda-feature-pipeline.h"
#include "feat/wave-reader.h"

namespace kaldi {
namespace cuda_decoder {
struct BatchedThreadedNnet3CudaPipelineConfig {
  BatchedThreadedNnet3CudaPipelineConfig() : use_online_ivectors(false) {}
  BatchedThreadedNnet3CudaOnlinePipelineConfig cuda_online_pipeline_opts;
  bool use_online_ivectors;
  void Register(OptionsItf *po) { cuda_online_pipeline_opts.Register(po); }
};

class BatchedThreadedNnet3CudaPipeline {
  const BatchedThreadedNnet3CudaPipelineConfig &config_;
  BatchedThreadedNnet3CudaOnlinePipeline cuda_online_pipeline_;
  using CorrelationID = BatchedThreadedNnet3CudaOnlinePipeline::CorrelationID;

  struct UtteranceTask {
    UtteranceTask &operator=(const UtteranceTask &) = delete;
    UtteranceTask(const UtteranceTask &) = delete;
    UtteranceTask(UtteranceTask &&) = default;
    UtteranceTask &operator=(UtteranceTask &&) = default;
    UtteranceTask() = default;

    std::shared_ptr<WaveData> wave_data;
    std::unique_ptr<SubVector<BaseFloat>>
        h_wave;  // (task.wave_data->Data(), 0)
    std::string key;
    int32 samp_offset;
    CorrelationID corr_id;
    std::atomic<int> *group_cnt;
    std::function<void(CompactLattice &)> callback;
    bool auto_close_after_callback;

    std::unique_ptr<CuMatrix<BaseFloat>>
        d_features;  // Used only when use_online_ivectors == false
    std::unique_ptr<CuVector<BaseFloat>>
        d_ivectors;  // Used only when use_online_ivectors == false
  };

  bool use_online_ivectors_;
  int n_input_per_chunk_;
  std::atomic<uint64_t> corr_id_cnt_;

  // Tasks added to the queue, but not yet used
  std::queue<UtteranceTask> preprocessing_utt_queue_;
  std::mutex preprocessing_utt_queue_m_;
  std::queue<UtteranceTask> outstanding_utt_;
  std::mutex outstanding_utt_m_;

  // Tasks currently being decoded by the cuda pipeline
  std::vector<UtteranceTask> current_tasks_;

  // Contains the ID of the tasks that are being completed
  // (we are decoding their last chunk)
  std::vector<UtteranceTask> tasks_last_chunk_;

  // Batch sent to online pipeline
  std::vector<CorrelationID> batch_corr_ids_;
  std::vector<bool> batch_is_last_chunk_;
  // Used when use_online_ivectors_
  std::vector<SubVector<BaseFloat>> batch_wave_samples_;
  // Used when !use_online_ivectors_
  std::vector<BaseFloat *> batch_features_;
  int batch_features_frame_stride_;
  std::vector<BaseFloat *> batch_ivectors_;
  std::vector<int> batch_n_input_frames_valid_;

  int32 max_batch_size_;
  // Thread responsible of feeding the online pipeline
  bool threads_running_;
  std::thread online_pipeline_control_thread_;

  // Number of tasks currently running
  std::atomic<int> n_tasks_not_done_;

  // Number of tasks currently running (per group)
  std::unordered_map<std::string, std::unique_ptr<std::atomic<int>>>
      n_group_tasks_not_done_;
  std::mutex n_group_tasks_not_done_m_;

  // If auto_close_after_callback is false, we will store the completed
  // lattices
  // there
  // They will be explicitely deleted by CloseDecodeHandle
  struct Output {
    Output() : is_clat_set(false) {}
    std::atomic<bool> is_clat_set;  // using a separate atomic because
                                    // std::atomic<std::shared_ptr> only exists
                                    // with C++20
    std::shared_ptr<CompactLattice> clat;
  };
  std::unordered_map<std::string, std::unique_ptr<Output>> completed_lattices_;
  std::mutex completed_lattices_m_;

  std::unique_ptr<OnlineCudaFeaturePipeline> cuda_features_;

  struct HostDeviceVector {
    cudaEvent_t evt;
    BaseFloat *h_data;
    BaseFloat *d_data;
    size_t size;

    // TODO move values to const
    HostDeviceVector() : h_data(NULL), d_data(NULL), size(16000 * 50) {
      cudaEventCreate(&evt);
      Reallocate(size);
    }

    virtual ~HostDeviceVector() {
      Deallocate();
      cudaEventDestroy(evt);
    }

    void Reallocate(size_t new_size) {
      KALDI_ASSERT(new_size > 0);
      Deallocate();
      cudaMalloc(&d_data, new_size * sizeof(*d_data));
      cudaMallocHost(&h_data, new_size * sizeof(*d_data));
      new_size = size;
    }
    void Deallocate() {
      if (d_data) cudaFree(d_data);
      if (h_data) cudaFreeHost(h_data);
    }
  };

  std::unique_ptr<HostDeviceVector> wave_buffer_, next_wave_buffer_;

 public:
  BatchedThreadedNnet3CudaPipeline(
      const BatchedThreadedNnet3CudaPipelineConfig &config,
      const fst::Fst<fst::StdArc> &decode_fst,
      const nnet3::AmNnetSimple &am_nnet, const TransitionModel &trans_model)
      : config_(config),
        cuda_online_pipeline_(config.cuda_online_pipeline_opts, decode_fst,
                              am_nnet, trans_model),
        use_online_ivectors_(config_.use_online_ivectors),
        corr_id_cnt_(0),
        max_batch_size_(config_.cuda_online_pipeline_opts.max_batch_size),
        threads_running_(true),
        online_pipeline_control_thread_(
            &BatchedThreadedNnet3CudaPipeline::ComputeTasks, this),
        n_tasks_not_done_(0) {
    batch_corr_ids_.reserve(max_batch_size_);
    batch_wave_samples_.reserve(max_batch_size_);
    batch_is_last_chunk_.reserve(max_batch_size_);
    tasks_last_chunk_.reserve(max_batch_size_);
    if (use_online_ivectors_) {
      n_input_per_chunk_ = cuda_online_pipeline_.GetNSampsPerChunk();
    } else {
      n_input_per_chunk_ = cuda_online_pipeline_.GetNInputFramesPerChunk();
      cuda_features_.reset(new OnlineCudaFeaturePipeline(
          config_.cuda_online_pipeline_opts.feature_opts));
      wave_buffer_.reset(new HostDeviceVector());
      next_wave_buffer_.reset(new HostDeviceVector());
    }
  }

  ~BatchedThreadedNnet3CudaPipeline() {
    threads_running_ = false;
    online_pipeline_control_thread_.join();
  }

  // Will decode wave_data. Then when done, will call the callback with the
  // final lattice.
  // It does not create a handle, so you don't need to call CloseDecodeHandle,
  // and GetLattice cannot be used with DecodeWithCallback (the lattice is
  // provided through the callback)
  // Should be preferred to OpenDecodeHandle/GetLattice/CloseDecodeHandle when
  // possible
  // The callback function is called in a multithreaded environment. It must be
  // threadsafe
  // To wait for those tasks to complete you can use WaitForGroup or
  // WaitForAllTasks
  void DecodeWithCallback(const std::shared_ptr<WaveData> &wave_data,
                          const std::function<void(CompactLattice &)> &callback,
                          const std::string &group = std::string()) {
    OpenDecodeHandle(std::string(), wave_data,
                     std::unique_ptr<SubVector<BaseFloat>>(), callback, true,
                     group);
  }

  void DecodeWithCallback(const VectorBase<BaseFloat> &wave_data,
                          float sample_rate,
                          const std::function<void(CompactLattice &)> &callback,
                          const std::string &group = std::string()) {
    KALDI_ASSERT(sample_rate == cuda_online_pipeline_.GetModelFrequency());
    std::unique_ptr<SubVector<BaseFloat>> h_wave(
        new SubVector<BaseFloat>(wave_data, 0, wave_data.Dim()));
    OpenDecodeHandle(std::string(), std::shared_ptr<WaveData>(),
                     std::move(h_wave), callback, true, group);
  }

  // Opens a new decode handle. The API user must take care of retriving the
  // final Lattice with GetLattice and close the handle with CloseDecodeHandle
  void OpenDecodeHandle(const std::string &key,
                        const std::shared_ptr<WaveData> &wave_data,
                        const std::string &group = std::string()) {
    OpenDecodeHandle(key, wave_data, std::unique_ptr<SubVector<BaseFloat>>(),
                     std::function<void(CompactLattice &)>(), false, group);
  }

  void OpenDecodeHandle(const std::string &key,
                        const VectorBase<BaseFloat> &wave_data,
                        float sample_rate,
                        const std::string &group = std::string()) {
    KALDI_ASSERT(sample_rate == cuda_online_pipeline_.GetModelFrequency());
    std::unique_ptr<SubVector<BaseFloat>> h_wave(
        new SubVector<BaseFloat>(wave_data, 0, wave_data.Dim()));
    OpenDecodeHandle(key, std::shared_ptr<WaveData>(), std::move(h_wave),
                     std::function<void(CompactLattice &)>(), false, group);
  }
  // Only necessary if OpenDecodeHandle was used. Not if DecodeWithCallback()
  // was used.
  void CloseDecodeHandle(const std::string &key);
  std::shared_ptr<CompactLattice> GetLattice(const std::string &key);
  // Will be depreciated
  bool GetLattice(const std::string &key, CompactLattice *lat) {
    std::shared_ptr<CompactLattice> shlat = GetLattice(key);
    *lat = *shlat;
    return true;
  }

  // Create a Task Group. Tasks can be associated with a group.
  // It is then possible to sync only on those tasks using WaitForGroup
  // (instead of WaitForAllTasks)
  void CreateTaskGroup(const std::string &group);
  void DestroyTaskGroup(const std::string &group);
  // Wait for all tasks in that group to complete
  void WaitForGroup(const std::string &group);

  void WaitForAllTasks();

  // Used for debug
  void SetSymbolTable(fst::SymbolTable *word_syms) {
    cuda_online_pipeline_.SetSymbolTable(word_syms);
  }

 private:
  // Private version of OpenDecodeHandle.
  // The public one ensure that the user doesn't use a key when
  // auto_close_after_callback is true
  void OpenDecodeHandle(const std::string &key,
                        const std::shared_ptr<WaveData> &wave_data,
                        std::unique_ptr<SubVector<BaseFloat>> &&h_wave,
                        const std::function<void(CompactLattice &)> &callback,
                        bool auto_close_after_callback,
                        const std::string &group = std::string());
  void BuildBatchFromCurrentTasks();
  void AcquireTasks();
  void ComputeTasks();
  void ComputeOfflineFeatures();
};

}  // end namespace cuda_decoder
}  // end namespace kaldi.

#endif  // KALDI_CUDA_DECODER_BATCHED_THREADED_CUDA_DECODER_H_
#endif  // HAVE_CUDA
