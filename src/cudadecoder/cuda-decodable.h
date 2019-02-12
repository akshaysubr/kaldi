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
#include "nnet3/decodable-simple-looped.h"
#include "nnet3/decodable-online-looped.h"
#include "lat/determinize-lattice-pruned.h"
#include <atomic>
#include <thread>

namespace kaldi {

	/* BatchedCudaDecoderConfig
	 * This class is a common configuration class for the various components
	 * of a batched cuda multi-threaded pipeline.  It defines a single place
	 * to control all operations and ensures that the various componets
	 * match configurations
	 */
	//configuration options common to the BatchedCudaDecoder and BatchedCudaDecodable
	struct BatchedCudaDecoderConfig {
		BatchedCudaDecoderConfig() : max_batch_size_(10), num_threads_(7), determinize_lattice_(true), max_pending_tasks_(2000) {};
		void Register(OptionsItf *po) {
			feature_opts_.Register(po);
			decodable_opts_.Register(po);
			decoder_opts_.Register(po);
			po->Register("max-batch-size",&max_batch_size_, "The maximum batch size to be used by the decoder.");
			po->Register("cuda-cpu-threads",&num_threads_, "The number of workpool threads to use in the cuda decoder");
      po->Register("determinize-lattice", &determinize_lattice_, "Determinize the lattice before output.");
      po->Register("max-outstanding-queue-length", &max_pending_tasks_, 
            "Number of files to allow to be outstanding at a time.  When the number of files is larger than this handles will be closed before opening new ones in FIFO order.");
			decoder_opts_.nlanes=max_batch_size_;
			decoder_opts_.nchannels=max_batch_size_;
      det_opts_.Register(po);

		}
		int max_batch_size_;
		int num_threads_;
    bool determinize_lattice_;
    int max_pending_tasks_;

		OnlineNnet2FeaturePipelineConfig  feature_opts_;           //constant readonly
		nnet3::NnetSimpleLoopedComputationOptions decodable_opts_; //constant readonly
		CudaDecoderConfig decoder_opts_;                           //constant readonly
    fst::DeterminizeLatticePhonePrunedOptions det_opts_;                 //constant readonly
	};

	class DecodableAmNnetLoopedOnlineCuda: public nnet3::DecodableNnetLoopedOnlineBase, public CudaDecodableInterface  {
		public:
			DecodableAmNnetLoopedOnlineCuda(
					const nnet3::DecodableNnetSimpleLoopedInfo &info,
					OnlineFeatureInterface *input_features,
					OnlineFeatureInterface *ivector_features) :  DecodableNnetLoopedOnlineBase(info, input_features, ivector_features)  {};

			~DecodableAmNnetLoopedOnlineCuda() {};

			//returns cuda pointer to nnet3 output
			virtual BaseFloat* GetLogLikelihoodsCudaPointer(int32 subsampled_frame) {
				EnsureFrameIsComputed(subsampled_frame);
				cudaStreamSynchronize(cudaStreamPerThread);      

				BaseFloat *frame_nnet3_out = current_log_post_.Data()+(subsampled_frame-current_log_post_subsampled_offset_)*current_log_post_.Stride();
				return frame_nnet3_out;
			};

			//DecodableInterface that should never be called.  Is inherited from DecodableNnetLoopedOnlineBase
			virtual int32 NumIndices() const { KALDI_ASSERT(false); return 0; } 
			virtual BaseFloat LogLikelihood(int32 subsampled_frame, int32 transition_id) { KALDI_ASSERT(false); return 0; } 

			virtual bool IsLastFrame(int32 subsampled_frame) const { return  nnet3::DecodableNnetLoopedOnlineBase::IsLastFrame(subsampled_frame); }
			virtual int32 NumFramesReady() const { return nnet3::DecodableNnetLoopedOnlineBase::NumFramesReady(); }

		private:
			KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmNnetLoopedOnlineCuda);
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
      struct TaskState {
        Vector<BaseFloat> raw_data; // Wave input data when wave_reader passed
        SubVector<BaseFloat> *wave_samples; // Used as a pointer to either the raw data or the samples passed
        std::string key;
        float sample_frequency;
        bool error;
        std::string error_string;

        Lattice lat;          //Raw Lattice output 
        CompactLattice dlat;  //Determinized lattice output.  Only set if determinize-lattice=true
        std::atomic<bool> finished;  //Tells master thread if task has finished execution

        TaskState() : wave_samples(NULL),sample_frequency(0), error(false), finished(false) {}
        ~TaskState() { delete wave_samples;}
        void Init(const std::string &key_in, const WaveData &wave_data_in) {
          raw_data.Resize(wave_data_in.Data().NumRows()*wave_data_in.Data().NumCols(), kUndefined);
          memcpy(raw_data.Data(), wave_data_in.Data().Data(), raw_data.Dim()*sizeof(BaseFloat));
          wave_samples=new SubVector<BaseFloat>(raw_data, 0, raw_data.Dim());
          sample_frequency=wave_data_in.SampFreq();
          finished=false;
          key=key_in;
        };
        void Init(const std::string &key_in, const VectorBase<BaseFloat> &wave_data_in, float sample_rate) {
          wave_samples=new SubVector<BaseFloat>(wave_data_in, 0, wave_data_in.Dim());
          sample_frequency =sample_rate;
          finished=false;
          key=key_in;
        }
      };

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

      std::map<std::string,TaskState> tasks_lookup_; //Contains a map of utterance to TaskState
      std::vector<std::thread> thread_contexts_;     //A list of thread contexts
	};



} // end namespace kaldi.


#endif

#endif
