// online2bin/online2-wav-nnet3-latgen-faster.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)
//           2016  Api.ai (Author: Ilya Platonov)

// See ../../COPYING for clarification regarding multiple authors
//
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

#include "feat/wave-reader.h"
#include "cudamatrix/cu-allocator.h"
#include "decoder/cuda-decoder.h"
#include "nnet3/decodable-simple-looped.h"
#include "nnet3/decodable-online-looped.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-thread.h"
#include "nnet3/nnet-utils.h"
#include <nvToolsExt.h>
#include <cuda_profiler_api.h>
#include <cuda.h>
#include <atomic>
#include <thread>
#include <chrono>

//#define REPLICATE_IN_BATCH  

using namespace kaldi;

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



 /* BatchedCudaDecoderConfig
 * This class is a common configuration class for the various components
 * of a batched cuda multi-threaded pipeline.  It defines a single place
 * to control all operations and ensures that the various componets
 * match configurations
 */
//configuration options common to the BatchedCudaDecoder and BatchedCudaDecodable
struct BatchedCudaDecoderConfig {
    BatchedCudaDecoderConfig() : max_batch_size_(20) {};
    void Register(ParseOptions *po) {
      feature_opts_.Register(po);
      decodable_opts_.Register(po);
      decoder_opts_.Register(po);
      po->Register("max-batch-size",&max_batch_size_, "The maximum batch size to be used by the decoder.");
      po->Register("num-threads",&num_threads_, "The number of workpool threads to use in the ThreadedBatchedCudaDecoder");
      decoder_opts_.nlanes=max_batch_size_;
      decoder_opts_.nchannels=max_batch_size_;

    }
    int max_batch_size_;
    int num_threads_;
    
    OnlineNnet2FeaturePipelineConfig  feature_opts_;           //constant readonly
    nnet3::NnetSimpleLoopedComputationOptions decodable_opts_; //constant readonly
    CudaDecoderConfig decoder_opts_;                           //constant readonly
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
 *    decoder.GetBestPath(utt_key,&lat);
 *    decoder.CloseDecodeHandle(utt_key);
 *
 *  decoder.Finalize();
 */
class ThreadedBatchedCudaDecoder {
  public:

    ThreadedBatchedCudaDecoder(const BatchedCudaDecoderConfig &config) : config_(config), max_pending_tasks_(2000) {};
    
    void Register(ParseOptions &po) {
      po.Register("max-outstanding-queue-length", &max_pending_tasks_, 
      "Number of files to allow to be outstanding at a time.  When the number of files is larger than this handles will be closed before opening new ones in FIFO order.");
    }

    //TODO should this take an nnet instead of a string?
    //allocates reusable objects that are common across all decodings
    void Initialize(const fst::Fst<fst::StdArc> &decode_fst, std::string nnet3_rxfilename) {
      KALDI_LOG << "ThreadedBatchedCudaDecoder Initialize with " << config_.num_threads_ << " threads\n";
   
     //read transition model and nnet
      bool binary;
      Input ki(nnet3_rxfilename, &binary);
      trans_model_.Read(ki.Stream(), binary);
      am_nnet_.Read(ki.Stream(), binary);
      SetBatchnormTestMode(true, &(am_nnet_.GetNnet()));
      SetDropoutTestMode(true, &(am_nnet_.GetNnet()));
      nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet_.GetNnet()));

      cuda_fst_.Initialize(decode_fst, trans_model_); 

      decodable_info_=new nnet3::DecodableNnetSimpleLoopedInfo(config_.decodable_opts_,&am_nnet_);

      //initialize threads and save their contexts so we can join them later
      thread_contexts_.resize(config_.num_threads_);

      //create work queue
      pending_task_queue_ = new TaskState*[max_pending_tasks_+1]; 
      tasks_front_ =0;
      tasks_back_ =0;

      //ensure all allocations/kernels above are complete before launching threads in different streams.
      cudaStreamSynchronize(cudaStreamPerThread);

      exit_=false;
      numStarted_=0;
      //start workers
      for (int i=0;i<config_.num_threads_;i++) {
        thread_contexts_[i]=std::thread(&ThreadedBatchedCudaDecoder::ExecuteWorker,this,i);
      }

      //wait for threads to start to ensure allocation time isn't in the timings
      while (numStarted_<config_.num_threads_);

    }
    void Finalize() {

      //Tell threads to exit and join them
      exit_=true;

      for (int i=0;i<config_.num_threads_;i++) {
        thread_contexts_[i].join();
      }

      cuda_fst_.Finalize();

      delete[] pending_task_queue_;
      delete decodable_info_;
    }

    //query a specific key to see if compute on it is complete
    bool isFinished(const std::string &key) {
      auto it=tasks_lookup_.find(key);
      KALDI_ASSERT(it!=tasks_lookup_.end());
      return it->second.finished;
    }

    //remove an audio file from the decoding and clean up resources
    void CloseDecodeHandle(const std::string &key) {
      auto it=tasks_lookup_.find(key);
      KALDI_ASSERT(it!=tasks_lookup_.end());

      TaskState &state = it->second;

      //wait for task to finish processing
      while (state.finished!=true);

      tasks_lookup_.erase(it);
    }


    //Adds a decoding task to the decoder
    bool OpenDecodeHandle(const std::string &key, const WaveData &wave_data) {

      //If no room for another task return false
      if(NumPendingTasks()==max_pending_tasks_)
        return false;

      //ensure key is unique
      KALDI_ASSERT(tasks_lookup_.end()==tasks_lookup_.find(key));

      //Create a new task in lookup map
      TaskState* t=&tasks_lookup_[key];
      t->Init(wave_data); 

      //Should not have changed so just doing a sanity check
      KALDI_ASSERT(NumPendingTasks()<max_pending_tasks_);

      //insert into pending task queue
      //locking should not be necessary as only the master thread writes to the queue and tasks_back_.  
      pending_task_queue_[tasks_back_]=t;
      //printf("New task: %p:%s, loc: %d\n", t, key.c_str(), (int)tasks_back_);
      tasks_back_=(tasks_back_+1)%(max_pending_tasks_+1);
      return true;
    }

    void GetBestPath(const std::string &key, Lattice *lat) {
      auto it=tasks_lookup_.find(key);
      KALDI_ASSERT(it!=tasks_lookup_.end());

      TaskState *state = &it->second;

      //wait for task to finish.  This should happens automatically without intervention from the master thread.
      while (state->finished==false);

      //Store off the lattice
      *lat=state->lat;
    }
    
    inline int NumPendingTasks() {
      return (tasks_back_ - tasks_front_ + max_pending_tasks_+1) % (max_pending_tasks_+1); 
    };

  private:

    //State needed for each decode task.  
    struct TaskState {
      WaveData wave_data;   //Wave data input
      Lattice lat;          //Lattice output
      std::atomic<bool> finished;  //Tells master thread if task has finished execution

      TaskState() : finished(false) {};
      void Init(const WaveData &wave_data_in) { wave_data=wave_data_in; finished=false; };
    };

    void ExecuteWorker(int threadId) {
      //Initialize this threads device
      CuDevice::Instantiate();

      //reusable across decodes
      OnlineNnet2FeaturePipelineInfo feature_info(config_.feature_opts_);
      feature_info.ivector_extractor_info.use_most_recent_ivector = true;
      feature_info.ivector_extractor_info.greedy_ivector_extractor = true;
      CudaDecoder cuda_decoders(cuda_fst_,config_.decoder_opts_,config_.max_batch_size_,config_.max_batch_size_);

      //This threads task list
      std::vector<TaskState*> tasks;
      //channel vectors
      std::vector<ChannelId> channels;        //active channels
      std::vector<ChannelId> free_channels;   //channels that are inactive
      std::vector<ChannelId> init_channels;   //channels that have yet to be initialized

      //channel state vectors
      std::vector<BaseFloat> samp_freqs;
      std::vector<SubVector<BaseFloat>* > data;
      std::vector<OnlineNnet2FeaturePipeline*> features;
      std::vector<CudaDecodableInterface*> decodables;
      std::vector<int> completed_channels;         
      std::vector<Lattice*> lattices;        

      tasks.reserve(config_.max_batch_size_);
      channels.reserve(config_.max_batch_size_);
      free_channels.reserve(config_.max_batch_size_);
      init_channels.reserve(config_.max_batch_size_);
      samp_freqs.reserve(config_.max_batch_size_);
      data.reserve(config_.max_batch_size_);
      features.reserve(config_.max_batch_size_);
      decodables.reserve(config_.max_batch_size_);
      completed_channels.reserve(config_.max_batch_size_);
      lattices.reserve(config_.max_batch_size_);
      
      //add all channels to free channel list
      for (int i=0;i<config_.max_batch_size_;i++) {
        free_channels.push_back(i);
      }      

      numStarted_++;  //Tell master I have started

      //main control loop.  Check if master has asked us to exit.
      while (!exit_) {

        do {  //processing loop.  Always run this at least once to try to grab work.
          //attempt to fill the batch
          if (tasks_front_!=tasks_back_)  { //if work is available grab more work

            int tasksRequested= free_channels.size();      
            int start=tasks.size(); 

#ifdef REPLICATE_IN_BATCH
            KALDI_ASSERT(tasksRequested==config_.max_batch_size_);
            KALDI_ASSERT(start==0);
            //wait for the full batch to be created
            while(NumPendingTasks()<config_.max_batch_size_);
#endif

            tasks_mutex_.lock(); //lock required because front might change from other workers

            //compute number of tasks to grab
            int tasksAvailable = NumPendingTasks();
            int tasksAssigned = std::min(tasksAvailable, tasksRequested);

            if (tasksAssigned>0) {
              //grab tasks
              for (int i=0;i<tasksAssigned;i++) {
                //printf("%d, Assigned task[%d]: %p\n", i, (int)tasks_front_, pending_task_queue_[tasks_front_]);
                tasks.push_back(pending_task_queue_[tasks_front_]);
                tasks_front_=(tasks_front_+1)%(max_pending_tasks_+1);              
              }
            }

            tasks_mutex_.unlock();

#ifdef REPLICATE_IN_BATCH 
            KALDI_ASSERT(free_channels.size()==config_.max_batch_size_);
#endif
            //allocate new data structures.  New decodes are in the range of [start,tasks.size())
            for (int i=start;i<tasks.size();i++) {
              TaskState &state = *tasks[i];

              //assign a free channel
              ChannelId channel=free_channels.back();
              free_channels.pop_back();

              channels.push_back(channel);      //add channel to processing list
              init_channels.push_back(channel); //add new channel to initialization list

              //create decoding state
              OnlineNnet2FeaturePipeline *feature = new OnlineNnet2FeaturePipeline(feature_info);
              features.push_back(feature);

              decodables.push_back(new DecodableAmNnetLoopedOnlineCuda(*decodable_info_, feature->InputFeature(), feature->IvectorFeature()));
              data.push_back(new SubVector<BaseFloat>(state.wave_data.Data(), 0));
              samp_freqs.push_back(state.wave_data.SampFreq());

              //Accept waveforms
              feature->AcceptWaveform(samp_freqs[i],*data[i]);
              feature->InputFinished();
            }
          } //end if(tasks_front_!=tasks_back_)

          if (tasks.size()==0) {
            break;  //no work so exit loop and try to get more work
          } 

#ifdef REPLICATE_IN_BATCH
          KALDI_ASSERT(free_channels.size()==0);
          KALDI_ASSERT(init_channels.size()==config_.max_batch_size_);
          KALDI_ASSERT(channels.size()==config_.max_batch_size_);
#endif

          if (init_channels.size()>0) {  //Except for the first iteration the size of this is typically 1 and rarely 2.
            //init decoding on new channels_
            cuda_decoders.InitDecoding(init_channels);   
            init_channels.clear();
          }

          nvtxRangePushA("AdvanceDecoding");
          //Advance decoding on all open channels
          cuda_decoders.AdvanceDecoding(channels,decodables);
          nvtxRangePop();

          //reorder arrays to put finished at the end      
          int cur=0;     //points to the last unfinished decode
          int back=tasks.size()-1;  //points to the last unchecked decode

          completed_channels.clear();
          lattices.clear();

          for (int i=0;i<tasks.size();i++) {
            ChannelId channel=channels[cur];
            TaskState &state=*tasks[cur];
            int numDecoded=cuda_decoders.NumFramesDecoded(channel);
            int toDecode=decodables[cur]->NumFramesReady();

            if (toDecode==numDecoded) {  //if current task is completed  
              lattices.push_back(&state.lat);
              completed_channels.push_back(channel);
              free_channels.push_back(channel);

              //move last element to this location
              std::swap(tasks[cur],tasks[back]);
              std::swap(channels[cur],channels[back]);
              std::swap(decodables[cur],decodables[back]);
              std::swap(features[cur],features[back]);
              std::swap(samp_freqs[cur],samp_freqs[back]);
              std::swap(data[cur],data[back]); 

              //back full now so decrement it
              back--;
            } else { 
#ifdef REPLICATE_IN_BATCH
              KALDI_ASSERT(false);
#endif
              //not completed move to next task
              cur++;
            }  //end if completed[cur]
          } //end for loop

#ifdef REPLICATE_IN_BATCH
          KALDI_ASSERT(free_channels.size()==config_.max_batch_size_);
#endif

          //Get best path for completed tasks
          cuda_decoders.GetBestPath(completed_channels,lattices,true);

          //clean up
          for (int i=cur;i<tasks.size();i++) {
            delete decodables[i];
            delete features[i];
            delete data[i];
            tasks[i]->finished=true;
          }      

          tasks.resize(cur);
          channels.resize(cur);
          decodables.resize(cur);
          features.resize(cur);
          samp_freqs.resize(cur);
          data.resize(cur);

        } while (tasks.size()>0);  //more work to process don't check exit condition
      } //end while(!exit_)
    }  //end ExecuteWorker

    const BatchedCudaDecoderConfig &config_;

    int max_pending_tasks_; 

    CudaFst cuda_fst_;
    TransitionModel trans_model_;
    nnet3::AmNnetSimple am_nnet_;
    nnet3::DecodableNnetSimpleLoopedInfo *decodable_info_;

    std::mutex tasks_mutex_;                      //protects tasks_front_ and pending_task_queue_ for workers
    std::atomic<int> tasks_front_, tasks_back_;
    TaskState** pending_task_queue_;

    std::atomic<bool> exit_;                      //signals threads to exit
    std::atomic<int> numStarted_;                 //signals master how many threads have started

    std::map<std::string,TaskState> tasks_lookup_; //Contains a map of utterance to TaskState
    std::vector<std::thread> thread_contexts_;     //A list of thread contexts
};



void GetDiagnosticsAndPrintOutput(const std::string &utt,
                                  const fst::SymbolTable *word_syms,
                                  const CompactLattice &clat,
                                  int64 *tot_num_frames,
                                  double *tot_like) {
  if (clat.NumStates() == 0) {
    KALDI_WARN << "Empty lattice.";
    return;
  }
  CompactLattice best_path_clat;
  CompactLatticeShortestPath(clat, &best_path_clat);

  Lattice best_path_lat;
  ConvertLattice(best_path_clat, &best_path_lat);

  double likelihood;
  LatticeWeight weight;
  int32 num_frames;
  std::vector<int32> alignment;
  std::vector<int32> words;
  GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
  num_frames = alignment.size();
  likelihood = -(weight.Value1() + weight.Value2());
  *tot_num_frames += num_frames;
  *tot_like += likelihood;
  KALDI_VLOG(2) << "Likelihood per frame for utterance " << utt << " is "
                << (likelihood / num_frames) << " over " << num_frames
                << " frames.";

  if (word_syms != NULL) {
    std::cerr << utt << ' ';
    for (size_t i = 0; i < words.size(); i++) {
      std::string s = word_syms->Find(words[i]);
      if (s == "")
        KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
      std::cerr << s << ' ';
    }
    std::cerr << std::endl;
  }
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
      "Reads in wav file(s) and simulates online decoding with neural nets\n"
      "(nnet3 setup), with optional iVector-based speaker adaptation and\n"
      "optional endpointing.  Note: some configuration values and inputs are\n"
      "set via config files whose filenames are passed as options\n"
      "\n"
      "Usage: online2-wav-nnet3-latgen-faster [options] <nnet3-in> <fst-in> "
      "<wav-rspecifier> <lattice-wspecifier>\n";

    std::string word_syms_rxfilename;

    bool write_lattice = true;
    int num_todo = -1;
    int iterations=1;
    ParseOptions po(usage);
    int pipeline_length=2000; //length of pipeline of outstanding requests, this is independent of queue lengths in decoder

    po.Register("write-lattice",&write_lattice, "Output lattice to a file.  Setting to false is useful when benchmarking.");
    po.Register("word-symbol-table", &word_syms_rxfilename, "Symbol table for words [for debug output]");
    po.Register("file-limit", &num_todo, 
        "Limits the number of files that are processed by this driver.  After N files are processed the remaing files are ignored.  Useful for profiling.");
    po.Register("iterations", &iterations, "Number of times to decode the corpus.  Output will be written only once.");
    
    //Multi-threaded CPU and batched GPU decoder
    BatchedCudaDecoderConfig batchedDecoderConfig;

    RegisterCuAllocatorOptions(&po);
    batchedDecoderConfig.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      return 1;
    }
    
    g_cuda_allocator.SetOptions(g_allocator_options);
    CuDevice::Instantiate().SelectGpuId("yes");
    CuDevice::Instantiate().AllowMultithreading();
    
    ThreadedBatchedCudaDecoder CudaDecoder(batchedDecoderConfig);

    std::string nnet3_rxfilename = po.GetArg(1),
      fst_rxfilename = po.GetArg(2),
      wav_rspecifier = po.GetArg(3),
      clat_wspecifier = po.GetArg(4);

    CompactLatticeWriter clat_writer(clat_wspecifier);
    
    fst::Fst<fst::StdArc> *decode_fst= fst::ReadFstKaldiGeneric(fst_rxfilename);

    CudaDecoder.Initialize(*decode_fst, nnet3_rxfilename);

    delete decode_fst;

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_rxfilename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
        KALDI_ERR << "Could not read symbol table from file "
          << word_syms_rxfilename;

    int32 num_done = 0, num_err = 0;
    double tot_like = 0.0;
    int64 num_frames = 0;
    double total_audio=0;
    
    nvtxRangePush("Global Timer");
    auto start = std::chrono::high_resolution_clock::now(); //starting timer here so we can measure throughput without allocation overheads

    std::queue<std::string> processed;
    for (int iter=0;iter<iterations;iter++) {
      SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);

        for (; !wav_reader.Done(); wav_reader.Next()) {
          nvtxRangePushA("Utterance Iteration");

          std::string utt = wav_reader.Key();
          printf("Utterance: %s\n", utt.c_str());

          const WaveData &wave_data = wav_reader.Value();
          total_audio+=wave_data.Duration();

          while(!CudaDecoder.OpenDecodeHandle(utt,wave_data));
          processed.push(utt);
          num_done++;

#ifdef REPLICATE_IN_BATCH
          //HACK to replicate across batch, need to remove
          for (int i=1;i<batchedDecoderConfig.max_batch_size_;i++) {
            total_audio+=wave_data.Duration();
            num_done++;
            std::string key=utt+std::to_string(i);
            while(!CudaDecoder.OpenDecodeHandle(key,wave_data));
          }
#endif

         while (processed.size()>=pipeline_length) {
            std::string &utt = processed.front();
            Lattice lat;
            CompactLattice clat;
        
            CudaDecoder.GetBestPath(utt,&lat);
            ConvertLattice(lat, &clat);

            GetDiagnosticsAndPrintOutput(utt, word_syms, clat, &num_frames, &tot_like);

            if (write_lattice && iter==0 ) {
              clat_writer.Write(utt, clat);
            }
        
            CudaDecoder.CloseDecodeHandle(utt);

#ifdef REPLICATE_IN_BATCH
            //HACK to replicate across batch, need to remove
            for (int i=1;i<batchedDecoderConfig.max_batch_size_;i++) { 
              std::string key=utt+std::to_string(i);
              CudaDecoder.CloseDecodeHandle(key);
            }
#endif
            processed.pop();
          }

          nvtxRangePop();
          if (num_todo!=-1 && num_done>=num_todo) break;
        } //end utterance loop

      nvtxRangePushA("Lattice Write");
      while (processed.size()>0) {
        std::string &utt = processed.front();
        Lattice lat;
        CompactLattice clat;

        CudaDecoder.GetBestPath(utt,&lat);
        ConvertLattice(lat, &clat);

        GetDiagnosticsAndPrintOutput(utt, word_syms, clat, &num_frames, &tot_like);

        if (write_lattice && iter==0 ) {
          clat_writer.Write(utt, clat);
        }

        CudaDecoder.CloseDecodeHandle(utt);

#ifdef REPLICATE_IN_BATCH
        //HACK to replicate across batch, need to remove
        for (int i=1;i<batchedDecoderConfig.max_batch_size_;i++) { 
          std::string key=utt+std::to_string(i);
          CudaDecoder.CloseDecodeHandle(key);
        }
#endif
        processed.pop();

      } //end for
      nvtxRangePop();
      auto finish = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> total_time = finish-start;
      KALDI_LOG << "Iteration: " << iter << " Aggregate Total Time: " << total_time.count()
        << " Total Audio: " << total_audio 
        << " RealTimeX: " << total_audio/total_time.count() << std::endl;
    } //End iterations loop
    KALDI_LOG << "Decoded " << num_done << " utterances, "
      << num_err << " with errors.";
    KALDI_LOG << "Overall likelihood per frame was " << (tot_like / num_frames)
      << " per frame over " << num_frames << " frames.";

    delete word_syms; // will delete if non-NULL.

    clat_writer.Close();

    CudaDecoder.Finalize();  
    cudaDeviceSynchronize();

    auto finish = std::chrono::high_resolution_clock::now();
    nvtxRangePop();

    std::chrono::duration<double> total_time = finish-start;

    KALDI_LOG << "Overall: " << " Aggregate Total Time: " << total_time.count()
      << " Total Audio: " << total_audio 
      << " RealTimeX: " << total_audio/total_time.count() << std::endl;

    return 0;

    //return (num_done != 0 ? 0 : 1);
  } catch (const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
} // main()

