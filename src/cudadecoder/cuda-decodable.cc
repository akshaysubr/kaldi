// cudadecoder/cuda-decodable.cc
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

#if HAVE_CUDA == 1
#include "base/kaldi-utils.h"
#include "cudadecoder/cuda-decodable.h"
#include <nvToolsExt.h>

namespace kaldi {

  void ThreadedBatchedCudaDecoder::Initialize(const fst::Fst<fst::StdArc> &decode_fst, std::string nnet3_rxfilename) {
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

    feature_info_=new  OnlineNnet2FeaturePipelineInfo(config_.feature_opts_);
    feature_info_->ivector_extractor_info.use_most_recent_ivector = true;
    feature_info_->ivector_extractor_info.greedy_ivector_extractor = true;

    decodable_info_=new nnet3::DecodableNnetSimpleLoopedInfo(config_.decodable_opts_,&am_nnet_);

    //initialize threads and save their contexts so we can join them later
    thread_contexts_.resize(config_.num_threads_);

    //create work queue
    pending_task_queue_ = new TaskState*[config_.max_pending_tasks_+1]; 
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
  void ThreadedBatchedCudaDecoder::Finalize() {

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
  bool ThreadedBatchedCudaDecoder::isFinished(const std::string &key) {
    tasks_lookup_mutex_.lock();
    auto it=tasks_lookup_.find(key);
    tasks_lookup_mutex_.unlock();
    KALDI_ASSERT(it!=tasks_lookup_.end());
    return it->second.finished;
  }

  //remove an audio file from the decoding and clean up resources
  void ThreadedBatchedCudaDecoder::CloseDecodeHandle(const std::string &key) {
    tasks_lookup_mutex_.lock();
    auto it=tasks_lookup_.find(key);
    tasks_lookup_mutex_.unlock();
    KALDI_ASSERT(it!=tasks_lookup_.end());

    TaskState &state = it->second;

    //wait for task to finish processing
    while (state.finished!=true);

    tasks_lookup_mutex_.lock();
    tasks_lookup_.erase(it);
    tasks_lookup_mutex_.unlock();
  }


  //Adds a decoding task to the decoder
  void ThreadedBatchedCudaDecoder::OpenDecodeHandle(const std::string &key, const WaveData &wave_data) {

    //ensure key is unique
    tasks_lookup_mutex_.lock();
    KALDI_ASSERT(tasks_lookup_.end()==tasks_lookup_.find(key));

    //Create a new task in lookup map
    TaskState* t=&tasks_lookup_[key];
    tasks_lookup_mutex_.unlock();

    t->Init(key, wave_data); 

    tasks_add_mutex_.lock();
    
    //wait for room in pending task queue
    while (NumPendingTasks()==config_.max_pending_tasks_) {
        // qualfied to ensure the right call occurs on windows
        kaldi::Sleep(.01);
    }

    //insert into pending task queue
    //locking should not be necessary as only the master thread writes to the queue and tasks_back_.  
    pending_task_queue_[tasks_back_]=t;
    //printf("New task: %p:%s, loc: %d\n", t, key.c_str(), (int)tasks_back_);
    tasks_back_=(tasks_back_+1)%(config_.max_pending_tasks_+1);
    
    tasks_add_mutex_.unlock();
  }

  // Add a decoding task to the decoder with a passed array of samples
  void ThreadedBatchedCudaDecoder::OpenDecodeHandle(const std::string &key, const VectorBase<BaseFloat> &wave_data, float sample_rate)
  {
    //ensure key is unique
    tasks_lookup_mutex_.lock();
    KALDI_ASSERT(tasks_lookup_.end()==tasks_lookup_.find(key));

    //Create a new task in lookup map
    TaskState* t=&tasks_lookup_[key];
    tasks_lookup_mutex_.unlock();

    t->Init(key, wave_data, sample_rate);

    tasks_add_mutex_.lock();
    
    //wait for room in pending task queue
    while (NumPendingTasks()==config_.max_pending_tasks_) {
        // qualfied to ensure the right call occurs on windows
        kaldi::Sleep(.01);
    }

    //insert into pending task queue
    //locking should not be necessary as only the master thread writes to the queue and tasks_back_.  
    pending_task_queue_[tasks_back_]=t;
    //printf("New task: %p:%s, loc: %d\n", t, key.c_str(), (int)tasks_back_);
    tasks_back_=(tasks_back_+1)%(config_.max_pending_tasks_+1);

    tasks_add_mutex_.unlock();
  }

  bool ThreadedBatchedCudaDecoder::GetRawLattice(const std::string &key, Lattice *lat) {
    nvtxRangePushA("GetRawLattice");
    auto it=tasks_lookup_.find(key);
    KALDI_ASSERT(it!=tasks_lookup_.end());

    TaskState *state = &it->second;

    //wait for task to finish.  This should happens automatically without intervention from the master thread.
    while (state->finished==false);

    if(state->error) {
      nvtxRangePop();
      return false;
    }
    //Store off the lattice
    *lat=state->lat;
    nvtxRangePop();
    return true;
  }
  
  bool ThreadedBatchedCudaDecoder::GetLattice(const std::string &key, CompactLattice *clat) {
    nvtxRangePushA("GetLattice");
    auto it=tasks_lookup_.find(key);
    KALDI_ASSERT(it!=tasks_lookup_.end());

    TaskState *state = &it->second;

    //wait for task to finish.  This should happens automatically without intervention from the master thread.
    while (state->finished==false);
    
    if(state->error) {
      nvtxRangePop();
      return false;
    }

    if(!config_.determinize_lattice_) {
      //Determinzation was not done by worker threads so do it here
      //TODO make a copy because wrapper below is destructive.  There may be a better way to do this.  Ask Dan.
      Lattice lat=state->lat;
      DeterminizeLatticePhonePrunedWrapper(trans_model_, &lat, 
                    config_.decoder_opts_.lattice_beam, &state->dlat, config_.det_opts_);
    }
    *clat=state->dlat;    //grab compact lattice
    nvtxRangePop();
    return true;
  }

  void ThreadedBatchedCudaDecoder::ExecuteWorker(int threadId) {
    //Initialize this threads device
    CuDevice::Instantiate();

    //Data structures that are reusable across decodes but unique to each thread
    CudaDecoder cuda_decoders(cuda_fst_,config_.decoder_opts_,config_.max_batch_size_,config_.max_batch_size_);
    //This threads task list
    std::vector<TaskState*> tasks;
    //channel vectors
    std::vector<ChannelId> channels;        //active channels
    std::vector<ChannelId> free_channels;   //channels that are inactive
    std::vector<ChannelId> init_channels;   //channels that have yet to be initialized
    //channel state vectors
    std::vector<SubVector<BaseFloat>* > data;
    std::vector<OnlineNnet2FeaturePipeline*> features;
    std::vector<CudaDecodableInterface*> decodables;
    std::vector<int> completed_channels;         
    std::vector<Lattice*> lattices;               //Raw lattices
    std::vector<CompactLattice*> dlattices;       //Determinized Lattices

    //Initialize reuseale data structures
    {
      tasks.reserve(config_.max_batch_size_);
      channels.reserve(config_.max_batch_size_);
      free_channels.reserve(config_.max_batch_size_);
      init_channels.reserve(config_.max_batch_size_);
      data.reserve(config_.max_batch_size_);
      features.reserve(config_.max_batch_size_);
      decodables.reserve(config_.max_batch_size_);
      completed_channels.reserve(config_.max_batch_size_);
      lattices.reserve(config_.max_batch_size_);
      dlattices.reserve(config_.max_batch_size_);

      //add all channels to free channel list
      for (int i=0;i<config_.max_batch_size_;i++) {
        free_channels.push_back(i);
      }      
    }

    numStarted_++;  //Tell master I have started

    //main control loop.  At each iteration a thread will see if it has been asked to shut 
    //down.  If it has it will exit.  This loop condition will only be processed if all
    //other work assigned to this thread has been processed.
    while (!exit_) {

      //main processing loop.  At each iteration the thread will do the following:
      //1) Attempt to grab more work. 
      //2) Initialize any new work
      //3) Process work in a batch
      //4) Postprocess any completed work
      do {
        //1) attempt to fill the batch
        {
          if (tasks_front_!=tasks_back_)  { //if work is available grab more work

            int tasksRequested= free_channels.size();      
            int start=tasks.size(); 

            tasks_mutex_.lock(); //lock required because front might change from other workers

            //compute number of tasks to grab
            int tasksAvailable = NumPendingTasks();
            int tasksAssigned = std::min(tasksAvailable, tasksRequested);

            if (tasksAssigned>0) {
              //grab tasks
              for (int i=0;i<tasksAssigned;i++) {
                //printf("%d, Assigned task[%d]: %p\n", i, (int)tasks_front_, pending_task_queue_[tasks_front_]);
                tasks.push_back(pending_task_queue_[tasks_front_]);
                tasks_front_=(tasks_front_+1)%(config_.max_pending_tasks_+1);              
              }
            }

            tasks_mutex_.unlock();


            //allocate new data structures.  New decodes are in the range of [start,tasks.size())
            for (int i=start;i<tasks.size();i++) {
              TaskState &state = *tasks[i];

              //printf("%d: Key: %s\n", threadId, state.key.c_str());
              //assign a free channel
              ChannelId channel=free_channels.back();
              free_channels.pop_back();

              channels.push_back(channel);      //add channel to processing list
              init_channels.push_back(channel); //add new channel to initialization list

              //create decoding state
              OnlineNnet2FeaturePipeline *feature = new OnlineNnet2FeaturePipeline(*feature_info_);
              features.push_back(feature);

              decodables.push_back(new DecodableAmNnetLoopedOnlineCuda(*decodable_info_, feature->InputFeature(), feature->IvectorFeature()));

              data.push_back(new SubVector<BaseFloat>(*state.wave_samples, 0, state.wave_samples->Dim()));

              //Accept waveforms
              feature->AcceptWaveform(state.sample_frequency,*data[i]);
              feature->InputFinished();
            }
          } //end if(tasks_front_!=tasks_back_)
        } //end 1)

        if (tasks.size()==0) {
          break;  //no active work on this thread.  This can happen if another thread was assigned the work.
        } 

        try {
          //2) Initialize any new work by calling InitDecoding on init_channels.  
          //Must check if init_channels is non-zero because there may not always be new work.
          if (init_channels.size()>0) {  //Except for the first iteration the size of this is typically 1 and rarely 2.
            //init decoding on new channels_
            cuda_decoders.InitDecoding(init_channels);   
            init_channels.clear();
          }

          //3) Process outstanding work in a batch
          nvtxRangePushA("AdvanceDecoding");
          //Advance decoding on all open channels
          cuda_decoders.AdvanceDecoding(channels,decodables);
          nvtxRangePop();


          //4) Post process work.  This reorders completed work to the end,
          //copies results outs, and cleans up data structures
          {
            //reorder arrays to put finished at the end      
            int cur=0;     //points to the last unfinished decode
            int back=tasks.size()-1;  //points to the last unchecked decode

            completed_channels.clear();
            lattices.clear();
            dlattices.clear();

            for (int i=0;i<tasks.size();i++) {
              ChannelId channel=channels[cur];
              TaskState &state=*tasks[cur];
              int numDecoded=cuda_decoders.NumFramesDecoded(channel);
              int toDecode=decodables[cur]->NumFramesReady();

              if (toDecode==numDecoded) {  //if current task is completed  
                lattices.push_back(&state.lat);
                dlattices.push_back(&state.dlat);
                completed_channels.push_back(channel);
                free_channels.push_back(channel);

                //move last element to this location
                std::swap(tasks[cur],tasks[back]);
                std::swap(channels[cur],channels[back]);
                std::swap(decodables[cur],decodables[back]);
                std::swap(features[cur],features[back]);
                std::swap(data[cur],data[back]); 

                //back full now so decrement it
                back--;
              } else { 
                //not completed move to next task
                cur++;
              }  //end if completed[cur]
            } //end for loop

            //Get best path for completed tasks
            cuda_decoders.GetRawLattice(completed_channels,lattices,true);

            if(config_.determinize_lattice_) {
              nvtxRangePushA("DeterminizeLattice");
              for(int i=0;i<completed_channels.size();i++) {
                //TODO wrapper below is destructive and we want to save the raw lattice.  
                //So we make a copy here.  There may be a better way to do this.
                Lattice lat=*lattices[i]; 
                DeterminizeLatticePhonePrunedWrapper(trans_model_, &lat, 
                    config_.decoder_opts_.lattice_beam, dlattices[i], config_.det_opts_);
              }
              nvtxRangePop();
            }

            // clean up datastructures
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
            data.resize(cur);
          } //end 4) cleanup
        } catch (CudaDecoderException e) {
          if(!e.recoverable) {
            bool UNRECOVERABLE_EXCEPTION=false;
            KALDI_LOG << "Error unrecoverable cuda decoder error '" << e.what() << "'\n";
            KALDI_ASSERT(UNRECOVERABLE_EXCEPTION);
          } else {
            KALDI_LOG << "Error recoverable cuda decoder error '" << e.what() << "'\n";
            KALDI_LOG << "    Aborting batch for recovery.  Canceling the following decodes:\n";
            for(int i=0;i<tasks.size();i++) {
              ChannelId channel=channels[i];
              free_channels.push_back(channel);

              TaskState &state=*tasks[i];
              KALDI_LOG << "      Canceled: " << state.key << "\n";
              state.error=true;
              state.error_string=e.what();
              delete decodables[i];
              delete features[i];
              delete data[i];
              tasks[i]->finished=true;
            }
            tasks.resize(0);
            channels.resize(0);
            decodables.resize(0);
            features.resize(0);
            data.resize(0);
          }
        }
      } while (tasks.size()>0);  //more work to process don't check exit condition
    } //end while(!exit_)
  }  //end ExecuteWorker


} // end namespace kaldi.

#endif
