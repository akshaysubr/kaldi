// cudadecoder/batched-threaded-nnet3-cuda-online-pipeline.cc
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

#include "cudadecoder/batched-threaded-nnet3-cuda-online-pipeline.h"
#include <nvToolsExt.h>
#include "feat/feature-window.h"
#include "lat/lattice-functions.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace cuda_decoder {
void BatchedThreadedNnet3CudaOnlinePipeline::Initialize(
    const fst::Fst<fst::StdArc> &decode_fst) {
  ReadParametersFromModel();
  AllocateAndInitializeData(decode_fst);
}

void BatchedThreadedNnet3CudaOnlinePipeline::AllocateAndInitializeData(
    const fst::Fst<fst::StdArc> &decode_fst) {
  d_all_features_.Resize(max_batch_size_ * input_frames_per_chunk_, input_dim_,
                         kUndefined, kStrideEqualNumCols);
  d_all_ivectors_.Resize(
      max_batch_size_, ivector_dim_,
      kSetZero);  // TODO using kSetZero until ivector is ready
  h_all_waveform_.Resize(max_batch_size_, samples_per_chunk_, kUndefined,
                         kStrideEqualNumCols);
  cudaHostRegister(h_all_waveform_.Data(), h_all_waveform_.SizeInBytes(),
                   cudaHostRegisterDefault);
  d_all_waveform_.Resize(max_batch_size_, samples_per_chunk_, kUndefined,
                         kStrideEqualNumCols);

  d_all_log_posteriors_.Resize(max_batch_size_ * output_frames_per_chunk_,
                               trans_model_->NumPdfs(), kUndefined);
  available_channels_.resize(config_.num_channels);
  std::iota(available_channels_.begin(), available_channels_.end(),
            0);  // 0,1,2,3..
  corr_id2channel_.reserve(config_.num_channels);
  channel_frame_offset_.resize(config_.num_channels, 0);
  decodables_.reserve(max_batch_size_);
  for (int32 i = 0; i < max_batch_size_; ++i) {
    CuSubMatrix<BaseFloat> this_log_posteriors = d_all_log_posteriors_.RowRange(
        i * output_frames_per_chunk_, output_frames_per_chunk_);
    decodables_.emplace_back(*trans_model_, this_log_posteriors);
  }
  decoder_ichannels_.reserve(max_batch_size_);
  prev_decoder_ichannels_.reserve(max_batch_size_);
  decoder_decodables_.reserve(max_batch_size_);
  prev_decoder_decodables_.reserve(max_batch_size_);
  KALDI_ASSERT(config_.feature_opts.feature_type == "mfcc");
  cuda_features_extractor_.reset(new OnlineCudaBatchedFeaturePipeline(
      config_.feature_opts, max_batch_size_, 8000, config_.num_channels, 100,
      40));  // TODO params should be read from config_ in
             // OnlineCudaBatchedFeaturePipeline constructor
  cuda_fst_ = std::make_shared<CudaFst>();
  cuda_fst_->Initialize(decode_fst, trans_model_);
  cuda_decoder_.reset(new CudaDecoder(*cuda_fst_, config_.decoder_opts,
                                      max_batch_size_, config_.num_channels));
  cuda_decoder_->SetThreadPoolAndStartCPUWorkers(thread_pool_.get(), 4);
  n_samples_valid_.resize(max_batch_size_);
  n_input_frames_valid_.resize(max_batch_size_);
  n_output_frames_valid_.resize(max_batch_size_);
}

void BatchedThreadedNnet3CudaOnlinePipeline::SetLatticeCallback(
    CorrelationID corr_id,
    const std::function<void(CompactLattice &)> &callback) {
  std::unique_ptr<std::function<void(CompactLattice &)>> callback_ptr(
      new std::function<void(CompactLattice &)>(callback));

  std::lock_guard<std::mutex> lk(
		  corrid2callbacks_m_);
  bool inserted;
  std::tie(std::ignore, inserted) =
	  corrid2callbacks_.emplace(corr_id, std::move(callback_ptr));
  KALDI_ASSERT(inserted);
}

void BatchedThreadedNnet3CudaOnlinePipeline::InitCorrID(CorrelationID corr_id) {
  bool inserted;
  decltype(corr_id2channel_.end()) it;
  std::tie(it, inserted) =
      corr_id2channel_.insert({corr_id, -1});
  int32 ichannel;
  if(inserted) {
	  // The corr_id was not in use
	  std::lock_guard<std::mutex> lk(available_channels_m_);
	  KALDI_ASSERT(!available_channels_.empty());
	  ichannel = available_channels_.back();
	  available_channels_.pop_back();
	  it->second = ichannel;
  } else {
	  // This corr id was already in use but not closed 
	  // It can happen if for instance a channel lost connection and did not send its last chunk 
	  // Cleaning up 
	  KALDI_WARN << "This corr_id was already in use";
	  ichannel = it->second; 
	  std::lock_guard<std::mutex> lk(corrid2callbacks_m_);
	  corrid2callbacks_.erase(corr_id);
  }

  channel_frame_offset_[ichannel] = 0;
  list_channels_first_chunk_.push_back(ichannel);
  cuda_nnet3_->InitChannel(ichannel);
}
/*
void BatchedThreadedNnet3CudaOnlinePipeline::RunCUDAFeatureExtraction() {
  mfcc.ComputeBatchedFeatures(cu_waveform, samp_freq, vtln_warp, &cu_features);
}
*/
void BatchedThreadedNnet3CudaOnlinePipeline::DecodeBatch(
    const std::vector<CorrelationID> &corr_ids,
    const std::vector<SubVector<BaseFloat>> &wave_samples,
    const std::vector<bool> &end_of_sequence) {
  nvtxRangePushA("DecodeBatch");
  KALDI_ASSERT(corr_ids.size() > 0);
  KALDI_ASSERT(corr_ids.size() == wave_samples.size());
  KALDI_ASSERT(corr_ids.size() == end_of_sequence.size());

  nvtxRangePushA("H2H Wave");
  for (int i = 0; i < wave_samples.size(); ++i) {
    const SubVector<BaseFloat> &src = wave_samples[i];
    int size = src.Dim();
    if (end_of_sequence[i]) {
      if (size > 0)
        KALDI_WARN << "If end_of_sequence is set, the right context will be "
                      "flushed and used as chunk. wave_samples should be empty";
      n_samples_valid_[i] = 0;
      continue;
    }
    n_samples_valid_[i] = size;
    const BaseFloat *wave_src = src.Data();
    BaseFloat *wave_dst = h_all_waveform_.RowData(i);
    std::memcpy(wave_dst, wave_src, size * sizeof(BaseFloat));
  }
  nvtxRangePop();
  // CopyFromMat syncs, avoiding it
  KALDI_ASSERT(d_all_waveform_.SizeInBytes() == h_all_waveform_.SizeInBytes());
  cudaMemcpyAsync(d_all_waveform_.Data(), h_all_waveform_.Data(),
                  h_all_waveform_.SizeInBytes(), cudaMemcpyHostToDevice,
                  cudaStreamPerThread);

  std::vector<int> channels;
  ListIChannelsInBatch(corr_ids, &channels);
  KALDI_ASSERT(channels.size() == corr_ids.size());

  d_all_features_.Resize(max_batch_size_ * input_frames_per_chunk_, input_dim_,
                         kUndefined);
  std::vector<int32> start_of_utt(channels.size(), 0);
  std::vector<int32> end_of_utt(
      channels.size(),
      0);  // used because of type mismatch in cuda_features_extractor_

  std::set<int> new_ichannels;  // TODO clean that
  for (int32 ichannel : list_channels_first_chunk_)
    new_ichannels.insert(ichannel);
  for (int i = 0; i < end_of_sequence.size(); ++i) {
    end_of_utt[i] = end_of_sequence[i];
    start_of_utt[i] = new_ichannels.count(channels[i]);
  }

  d_all_features_.Resize(max_batch_size_ * input_frames_per_chunk_, input_dim_,
                         kUndefined, kStrideEqualNumCols);
  KALDI_ASSERT(cuda_features_extractor_);
  cuda_features_extractor_->ComputeFeatures(
      d_all_waveform_, model_frequency_, &d_all_features_, NULL, NULL, channels,
      start_of_utt, end_of_utt, n_samples_valid_, &n_input_frames_valid_);

  std::vector<BaseFloat *> d_features;
  std::vector<BaseFloat *> d_ivectors;
  for (int i = 0; i < corr_ids.size(); ++i) {
    d_features.push_back(d_all_features_.Data() +
                         i * input_frames_per_chunk_ *
                             d_all_features_.Stride());
    d_ivectors.push_back(d_all_ivectors_.Data() + i * d_all_ivectors_.Stride());
  }
  int features_frame_stride = d_all_features_.Stride();
  DecodeBatch(corr_ids, d_features, features_frame_stride,
              n_input_frames_valid_, d_ivectors, end_of_sequence, &channels);
}

void BatchedThreadedNnet3CudaOnlinePipeline::DecodeBatch(
    const std::vector<CorrelationID> &corr_ids,
    const std::vector<BaseFloat *> &d_features, const int features_frame_stride,
    const std::vector<int> &n_input_frames_valid,
    const std::vector<BaseFloat *> &d_ivectors,
    const std::vector<bool> &end_of_sequence, std::vector<int> *channels) {
  nvtxRangePushA("DecodeBatch");
  std::vector<int> buf;  // TODO
  if (!channels) {
    channels = &buf;
    ListIChannelsInBatch(corr_ids, channels);
  }
  // May already be filled before feature extraction (other overload of
  // DecodeBatch)
  //  if (curr_batch_ichannels_.empty()) ListIChannelsInBatch(batch);
  if (!list_channels_first_chunk_.empty())
    cuda_decoder_->InitDecoding(list_channels_first_chunk_);

  RunNnet3(*channels, d_features, features_frame_stride, n_input_frames_valid,
           d_ivectors, end_of_sequence);
  RunDecoder(*channels);

  BuildLatticesAndRunCallbacks(corr_ids, *channels, end_of_sequence);
  list_channels_first_chunk_.clear();
  nvtxRangePop();
}

void BatchedThreadedNnet3CudaOnlinePipeline::BuildLatticesAndRunCallbacks(
    const std::vector<CorrelationID> &corr_ids,
    const std::vector<int> &channels, const std::vector<bool> &is_last_chunk) {
  list_channels_last_chunk_.clear();
  list_corr_id_last_chunk_.clear();
  for (int i = 0; i < is_last_chunk.size(); ++i) {
    if (is_last_chunk[i]) {
      list_channels_last_chunk_.push_back(channels[i]);
      list_corr_id_last_chunk_.push_back(corr_ids[i]);
    }
  }
  cuda_decoder_->PrepareForGetRawLattice(list_channels_last_chunk_, true);
  // Storing number of callbacks not done. Used if  WaitForLatticeCallbacks()
  // is
  // called
  n_lattice_callbacks_not_done_.store(list_channels_last_chunk_.size());

  // delete data used for decoding that corr_id
  for (int32 i = 0; i < list_channels_last_chunk_.size(); ++i) {
    uint64_t ichannel = list_channels_last_chunk_[i];
    CorrelationID corr_id = list_corr_id_last_chunk_[i];
    int32 ndeleted = corr_id2channel_.erase(corr_id);
    KALDI_ASSERT(ndeleted == 1);
    KALDI_ASSERT(thread_pool_->tryPush(
        {&BatchedThreadedNnet3CudaOnlinePipeline::FinalizeDecodingWrapper, this,
         ichannel, corr_id}));
  }
  list_channels_last_chunk_.clear();
  list_corr_id_last_chunk_.clear();
}

void BatchedThreadedNnet3CudaOnlinePipeline::ListIChannelsInBatch(
    const std::vector<CorrelationID> &corr_ids, std::vector<int> *channels) {
  channels->clear();
  list_channels_last_chunk_.clear();
  list_corr_id_last_chunk_.clear();
  for (int i = 0; i < corr_ids.size(); ++i) {
    int corr_id = corr_ids[i];
    auto it = corr_id2channel_.find(corr_id);
    KALDI_ASSERT(it != corr_id2channel_.end());
    int ichannel = it->second;
    channels->push_back(ichannel);
  }
}

void BatchedThreadedNnet3CudaOnlinePipeline::RunNnet3(
    const std::vector<int> &channels,
    const std::vector<BaseFloat *> &d_features, const int features_stride,
    const std::vector<int> &n_input_frames_valid,
    const std::vector<BaseFloat *> &d_ivectors,
    const std::vector<bool> &end_of_sequence) {
  const std::vector<bool> &flush_context = end_of_sequence;
  cuda_nnet3_->RunBatch(channels, d_features, features_stride, d_ivectors,
                        n_input_frames_valid, flush_context,
                        &d_all_log_posteriors_, &n_output_frames_valid_);
}

void BatchedThreadedNnet3CudaOnlinePipeline::RunDecoder(
    const std::vector<int> &channels) {
  decoder_decodables_.clear();
  decoder_ichannels_.clear();
  for (int32 element = 0; element < channels.size(); ++element) {
    int32 ichannel = channels[element];
    decodables_[element].SetValidNRowsInMatrix(n_output_frames_valid_[element]);
    int32 previous_offset = channel_frame_offset_[ichannel];
    int32 n_output_frames = n_output_frames_valid_[element];
    channel_frame_offset_[ichannel] = previous_offset + n_output_frames;

    decodables_[element].SetFrameOffset(previous_offset);
    decoder_ichannels_.push_back(ichannel);
    decoder_decodables_.push_back(&decodables_[element]);
  }

  while (!decoder_ichannels_.empty()) {
    cuda_decoder_->AdvanceDecoding(decoder_ichannels_, decoder_decodables_);
    // Looking for utterances with more frames to process
    decoder_ichannels_.swap(prev_decoder_ichannels_);
    decoder_decodables_.swap(prev_decoder_decodables_);
    decoder_decodables_.clear();
    decoder_ichannels_.clear();
    for (int32 i = 0; i < prev_decoder_ichannels_.size(); ++i) {
      int32 ichannel = prev_decoder_ichannels_[i];
      DecodableCuMatrixMapped *decodable = prev_decoder_decodables_[i];
      bool is_done = (decodable->NumFramesReady() ==
                      cuda_decoder_->NumFramesDecoded(ichannel));
      if (!is_done) {
        decoder_ichannels_.push_back(ichannel);
        decoder_decodables_.push_back(decodable);
      }
    }
  }
}

void BatchedThreadedNnet3CudaOnlinePipeline::ReadParametersFromModel() {
  // TODO remove that
  feature_info_.reset(new OnlineNnet2FeaturePipelineInfo(config_.feature_opts));
  feature_info_->ivector_extractor_info.use_most_recent_ivector = true;
  feature_info_->ivector_extractor_info.greedy_ivector_extractor = true;

  OnlineNnet2FeaturePipeline feature(*feature_info_);
  // TODO clean following lines
  input_dim_ = feature.InputFeature()->Dim();
  ivector_dim_ = feature.IvectorFeature()->Dim();
  model_frequency_ = feature_info_->mfcc_opts.frame_opts.samp_freq;
  BaseFloat frame_shift = feature_info_->FrameShiftInSeconds();
  input_frames_per_chunk_ = config_.compute_opts.frames_per_chunk;
  seconds_per_chunk_ = input_frames_per_chunk_ * frame_shift;
  samples_per_chunk_ = seconds_per_chunk_ * model_frequency_;
  BatchedStaticNnet3Config nnet3_config;
  nnet3_config.compute_opts = config_.compute_opts;
  nnet3_config.max_batch_size = max_batch_size_;
  nnet3_config.nchannels = config_.num_channels;
  nnet3_config.has_ivector = true;
  cuda_nnet3_.reset(new BatchedStaticNnet3(nnet3_config, *am_nnet_));
  output_frames_per_chunk_ = cuda_nnet3_->GetNOutputFramesPerChunk();
}

void BatchedThreadedNnet3CudaOnlinePipeline::FinalizeDecoding(
    int32 ichannel, CorrelationID corr_id) {
  Lattice lat;
  cuda_decoder_->ConcurrentGetRawLatticeSingleChannel(ichannel, &lat);

  // Done with this channel. Making it available again
  {
    std::lock_guard<std::mutex> lk(available_channels_m_);
    available_channels_.push_back(ichannel);
  }

  // If necessary, determinize the lattice
  CompactLattice dlat;
  DeterminizeLatticePhonePrunedWrapper(*trans_model_, &lat,
                                       config_.decoder_opts.lattice_beam, &dlat,
                                       config_.det_opts);

  if (dlat.NumStates() == 0) {
    KALDI_WARN << "Empty lattice.";
  } else {
    if (word_syms_) {
      CompactLattice best_path_clat;
      CompactLatticeShortestPath(dlat, &best_path_clat);

      Lattice best_path_lat;
      ConvertLattice(best_path_clat, &best_path_lat);

      std::vector<int32> alignment;
      std::vector<int32> words;
      LatticeWeight weight;
      GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
      std::ostringstream oss;
      for (size_t i = 0; i < words.size(); i++) {
        std::string s = word_syms_->Find(words[i]);
        if (s == "") oss << "Word-id " << words[i] << " not in symbol table.";
        oss << s << " ";
      }
      {
        std::lock_guard<std::mutex> lk(stdout_m_);
        KALDI_LOG << "OUTPUT: " << oss.str();
      }
    }

    std::unique_ptr<std::function<void(CompactLattice &)>> callback;
    {
	    std::lock_guard<std::mutex> lk(corrid2callbacks_m_);
	    auto it =
		    corrid2callbacks_.find(corr_id);  // TODO remove map, use ichannel
	    if (it != corrid2callbacks_.end()) {
		    callback = std::move(it->second);
		    corrid2callbacks_.erase(it);
	    }

    }
    // if ptr set and if callback func callable
    if (callback && *callback) {
	    (*callback)(dlat);
    }

    // TODO erase it
  }
  n_lattice_callbacks_not_done_.fetch_sub(1);
}

void BatchedThreadedNnet3CudaOnlinePipeline::WaitForLatticeCallbacks() {
  while (n_lattice_callbacks_not_done_.load() != 0) {
    usleep(10000);  // TODO
  }
}

}  // end namespace cuda_decoder
}  // end namespace kaldi.

#endif  // HAVE_CUDA
