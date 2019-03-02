// cudadecoder/cuda-decodable.h
// TODO nvidia apache2
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#ifndef KALDI_CUDA_DECODEABLE_H_
#define KALDI_CUDA_DECODEABLE_H_

#if HAVE_CUDA == 1

#include "feat/wave-reader.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "cudadecoder/cuda-decoder.h"
#include "decoder/decodable-matrix.h"
#include "lat/determinize-lattice-pruned.h"
#include "nnet3/nnet-batch-compute.h"
#include <atomic>
#include <thread>
#include "ThreadPool.h"

namespace kaldi {

  /* BatchedCudaDecoderConfig
   * This class is a common configuration class for the various components
   * of a batched cuda multi-threaded pipeline.  It defines a single place
   * to control all operations and ensures that the various componets
   * match configurations
   */
  //configuration options common to the BatchedCudaDecoder and BatchedCudaDecodable
  struct BatchedCudaDecoderConfig {
    BatchedCudaDecoderConfig() : max_batch_size_(10), batch_drain_size_(5), 
    num_control_threads_(7), num_worker_threads_(4), determinize_lattice_(true), max_pending_tasks_(2000) {};
    void Register(OptionsItf *po) {

      po->Register("max-batch-size",&max_batch_size_, "The maximum batch size to be used by the decoder.");
      po->Register("batch-drain-size",&batch_drain_size_, "How far to drain the batch before refilling work.  This batches pre/post decode work.");
      po->Register("cuda-control-threads",&num_control_threads_, "The number of workpool threads to use in the cuda decoder");
      po->Register("cuda-worker-threads",&num_worker_threads_, "The number of sub threads a worker can spawn to help with CPU tasks.");
      po->Register("determinize-lattice", &determinize_lattice_, "Determinize the lattice before output.");
      po->Register("max-outstanding-queue-length", &max_pending_tasks_, 
          "Number of files to allow to be outstanding at a time.  When the number of files is larger than this handles will be closed before opening new ones in FIFO order.");

      decoder_opts_.nlanes=max_batch_size_;
      decoder_opts_.nchannels=max_batch_size_;

      feature_opts_.Register(po);
      decoder_opts_.Register(po);
      det_opts_.Register(po);
      compute_opts_.Register(po);
    }
    int max_batch_size_;
    int batch_drain_size_;
    int num_control_threads_;
    int num_worker_threads_;
    bool determinize_lattice_;
    int max_pending_tasks_;

    OnlineNnet2FeaturePipelineConfig  feature_opts_;           //constant readonly
    CudaDecoderConfig decoder_opts_;                           //constant readonly
    fst::DeterminizeLatticePhonePrunedOptions det_opts_;       //constant readonly
    nnet3::NnetBatchComputerOptions compute_opts_;             //constant readonly
  };

  /**
    Cuda Decodable matrix.  Takes transition model and posteriors and provides
    an interface similar to the Decodable Interface
    */
  class DecodableCuMatrixMapped: public CudaDecodableInterface {
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

      virtual ~DecodableCuMatrixMapped() {};

      //returns cuda pointer to nnet3 output
      virtual BaseFloat* GetLogLikelihoodsCudaPointer(int32 subsampled_frame);


    private:
      const TransitionModel &trans_model_;  // for tid to pdf mapping
      const CuMatrixBase<BaseFloat> *likes_;

      int32 frame_offset_;

      // raw_data_ and stride_ are a kind of fast look-aside for 'likes_', to be
      // used when KALDI_PARANOID is false.
      const BaseFloat *raw_data_;
      int32 stride_;

      KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableCuMatrixMapped);
  };

  /*
   *  ThreadedBatchedCudaDecoder uses multiple levels of parallelism in order to decode quickly on CUDA GPUs.
   *  It's API is utterance centric using deferred execution.  That is a user submits work one utterance at a time
   *  and the class batches that work behind the scene. Utterance are passed into the API with a unique key of type string.
   *  The user must ensure this name is unique.  APIs are provided to enqueue work, query the best path, and cleanup enqueued work.
   *  Once a user closes a decode handle they are free to use that key again.
   *  
   *  Example Usage is as follows:
   *  ThreadedBatchedCudaDecoder decoder;
   *  decoder.Initalize(decode_fst, am_nnet_rx_file);
   *   
   *  //some loop
   *    std::string utt_key = ...
   *    while(!decoder.OpenDecodeHandle(utt_key,wave_data));
   * 
   *  ...
   *
   *  //some loop
   *    Lattice lat;
   *    std::string utt_key = ...
   *    decoder.GetRawLattice(utt_key,&lat);
   *    decoder.CloseDecodeHandle(utt_key);
   *
   *  decoder.Finalize();
   */
  class ThreadedBatchedCudaDecoder {
    public:

      ThreadedBatchedCudaDecoder(const BatchedCudaDecoderConfig &config) : config_(config) {};

      //TODO should this take an nnet instead of a string?
      //allocates reusable objects that are common across all decodings
      void Initialize(const fst::Fst<fst::StdArc> &decode_fst, std::string nnet3_rxfilename);
      //deallocates reusable objects
      void Finalize();

      //query a specific key to see if compute on it is complete
      bool isFinished(const std::string &key);

      //remove an audio file from the decoding and clean up resources
      void CloseDecodeHandle(const std::string &key);

      //Adds a decoding task to the decoder
      void OpenDecodeHandle(const std::string &key, const WaveData &wave_data);
      // When passing in a vector of data, the caller must ensure the data exists until the CloseDecodeHandle is called
      void OpenDecodeHandle(const std::string &key, const VectorBase<BaseFloat> &wave_data, float sample_rate);

      //Copies the raw lattice for decoded handle "key" into lat
      bool GetRawLattice(const std::string &key, Lattice *lat);
      //Determinizes raw lattice and returns a compact lattice
      bool GetLattice(const std::string &key, CompactLattice *lat);

      inline int NumPendingTasks() {
        return (tasks_back_ - tasks_front_ + config_.max_pending_tasks_+1) % (config_.max_pending_tasks_+1); 
      };

    private:

      //State needed for each decode task. 
      //This state can be passed around by reference or pointer safely
      //and provides a convieniet way to store all decoding state.
      struct TaskState {
        Vector<BaseFloat> raw_data_; // Wave input data when wave_reader passed
        SubVector<BaseFloat> *wave_samples_; // Used as a pointer to either the raw data or the samples passed
        std::string key_;
        float sample_frequency_;
        bool error_;
        std::string error_string_;

        Lattice lat_;          //Raw Lattice output 
        CompactLattice dlat_;  //Determinized lattice output.  Only set if determinize-lattice=true
        std::atomic<bool> finished_;  //Tells master thread if task has finished execution

        bool determinized_;
        
        Vector<BaseFloat> ivector_features_;
        Matrix<BaseFloat> input_features_;
        CuMatrix<BaseFloat> posteriors_;

        TaskState() : wave_samples_(NULL), sample_frequency_(0), error_(false), finished_(false), determinized_(false) {}
        ~TaskState() { if(wave_samples_) delete wave_samples_;}

        //Init when wave data is passed directly in.  This data is deep copied.
        void Init(const std::string &key_in, const WaveData &wave_data_in) {
          raw_data_.Resize(wave_data_in.Data().NumRows()*wave_data_in.Data().NumCols(), kUndefined);
          memcpy(raw_data_.Data(), wave_data_in.Data().Data(), raw_data_.Dim()*sizeof(BaseFloat));
          wave_samples_=new SubVector<BaseFloat>(raw_data_, 0, raw_data_.Dim());
          sample_frequency_=wave_data_in.SampFreq();
          determinized_=false;
          finished_=false;
          key_=key_in;
        };
        //Init when raw data is passed in.  This data is shallow copied.
        void Init(const std::string &key_in, const VectorBase<BaseFloat> &wave_data_in, float sample_rate) {
          wave_samples_=new SubVector<BaseFloat>(wave_data_in, 0, wave_data_in.Dim());
          sample_frequency_=sample_rate;
          determinized_=false;
          finished_=false;
          key_=key_in;
        }
      };

      //Holds the current channel state for a worker
      struct ChannelState {
        std::vector<ChannelId> channels_;
        std::vector<ChannelId> free_channels_; 
        std::vector<ChannelId> completed_channels_; 
      };

      //Adds task to the PendingTaskQueue
      void AddTaskToPendingTaskQueue(TaskState *task);

      //Attempts to fill the batch from the task queue.  May not fully fill the batch.
      void AquireAdditionalTasks(CudaDecoder &cuda_decoder,
          ChannelState &channel_state,
          std::vector<TaskState*> &tasks);

      //Computes Features for a single decode instance.  
      void ComputeOneFeature(TaskState *task);

      //Computes Nnet across the current decode batch
      void ComputeBatchNnet(nnet3::NnetBatchComputer &computer, int32 first, 
          std::vector<TaskState*> &tasks);

      //Allocates decodables for tasks in the range of dstates[first,dstates.size())
      void AllocateDecodables(int32 first, std::vector<TaskState*> &tasks,
          std::vector<CudaDecodableInterface*> &decodables);

      //Removes all completed channels from the channel list.
      //Also enqueues up work for post processing
      void RemoveCompletedChannels(CudaDecoder &cuda_decoder,
          ChannelState &channel_state,
          std::vector<CudaDecodableInterface*> &decodables,
          std::vector<TaskState*> &tasks);

      //For each completed decode perform post processing work and clean up
      void PostDecodeProcessing(CudaDecoder &cuda_decoder,
          ChannelState &channel_state,
          std::vector<CudaDecodableInterface*> &decodables,
          std::vector<TaskState*> &tasks);

      void DeterminizeOneLattice(TaskState *state);

      //Thread execution function.  This is a single worker thread which processes input.
      void ExecuteWorker(int threadId);

      const BatchedCudaDecoderConfig &config_;

      CudaFst cuda_fst_;
      TransitionModel trans_model_;
      nnet3::AmNnetSimple am_nnet_;
      nnet3::DecodableNnetSimpleLoopedInfo *decodable_info_;
      OnlineNnet2FeaturePipelineInfo *feature_info_;

      std::mutex tasks_mutex_;                      //protects tasks_front_ and pending_task_queue_ for workers
      std::mutex tasks_add_mutex_;                  //protect OpenDecodeHandle if multiple threads access 
      std::mutex tasks_lookup_mutex_;               //protext tasks_lookup map 
      std::atomic<int> tasks_front_, tasks_back_;
      TaskState** pending_task_queue_;

      std::atomic<bool> exit_;                      //signals threads to exit
      std::atomic<int> numStarted_;                 //signals master how many threads have started

      ThreadPool *work_pool_;                      //thread pool for CPU work

      std::map<std::string,TaskState> tasks_lookup_; //Contains a map of utterance to TaskState
      std::vector<std::thread> thread_contexts_;     //A list of thread contexts
  };



} // end namespace kaldi.


#endif

#endif
