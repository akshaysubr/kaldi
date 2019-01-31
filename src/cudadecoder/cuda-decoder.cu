// decoder/cuda-decoder.cu
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

#include "cudadecoder/cuda-decoder.h"
#include <algorithm>
#include <nvToolsExt.h>
#include <cuda_runtime_api.h>
#include <float.h>
#include <algorithm>
#include <cub/cub.cuh>
#include "cuda-decoder-kernels.h"

#include <tuple>
#include <map>

#define MEMADVISE

namespace kaldi {
	CudaDecoder::CudaDecoder(const CudaFst &fst, 
			const CudaDecoderConfig &config,
			int32 nlanes,
			int32 nchannels): fst_(fst), 
	default_beam_(config.default_beam),
	lattice_beam_(config.lattice_beam),
	generate_lattices_(config.generate_lattices),
	max_tokens_(config.max_tokens), 
	max_tokens_per_frame_(config.max_tokens_per_frame),
	max_active_(config.max_active), 
	nlanes_(nlanes),
	nchannels_(nchannels) {
		KALDI_ASSERT(nlanes_ < KALDI_CUDA_DECODER_MAX_N_LANES);
		//
		// For a description of the class members, please refer to the cuda-decoder.h file
		//
		cudaStreamCreate(&compute_st_);
		cudaStreamCreate(&copy_st_); 

		cudaEventCreate(&can_read_h_main_q_narcs_);
		cudaEventCreate(&can_write_to_main_q_);
		cudaEventCreate(&can_read_final_h_main_q_end_);
		cudaEventCreate(&before_finalize_nonemitting_kernel_);

		cudaEventCreate(&can_use_acoustic_cost_);
		cudaEventCreate(&can_use_infotoken_);
		cudaEventCreate(&can_use_extra_cost_);

		KALDI_ASSERT(nlanes > 0);
		KALDI_ASSERT(nchannels > 0);

		++nchannels_; // allocating init_channel_params at the same time
		init_channel_id_ = nchannels_-1; // Using last one as init_channel_params
		hashmap_capacity_ = max_tokens_per_frame_; // TODO

		d_channels_counters_.Resize(nchannels_, 1);
		d_lanes_counters_.Resize(nlanes, 1);
		d_main_q_state_and_cost_.Resize(nchannels_, max_tokens_per_frame_);
		d_main_q_info_.Resize(nlanes, max_tokens_per_frame_);
		d_aux_q_state_and_cost_.Resize(nlanes, max_tokens_per_frame_);
		d_aux_q_info_.Resize(nlanes, max_tokens_per_frame_);
		d_main_q_degrees_prefix_sum_.Resize(nchannels_, max_tokens_per_frame_);
		d_histograms_.Resize(nlanes_, KALDI_CUDA_DECODER_NBINS);
		cudaMemsetAsync(d_histograms_.lane(0), 0, sizeof(int32)*KALDI_CUDA_DECODER_NBINS*nlanes_, compute_st_);

		// TODO use aux_q_state_and_cost for following 2		
		// TODO maybe aux_q_info because that one can be used temporarly 
		d_main_q_extra_prev_tokens_prefix_sum_.Resize(nlanes_, max_tokens_per_frame_);
		d_main_q_n_extra_prev_tokens_local_idx_.Resize(nlanes_, max_tokens_per_frame_);
		
		// TODO can use aux_q_acoustic for following
		d_main_q_representative_id_.Resize(nlanes_, max_tokens_per_frame_); 

		// +8
		d_main_q_extra_prev_tokens_.Resize(nlanes_, max_tokens_per_frame_); 
		// +8
		d_main_q_extra_cost_.Resize(nlanes_, max_tokens_per_frame_);

		d_main_q_block_sums_prefix_sum_.Resize(nlanes, 
				KALDI_CUDA_DECODER_DIV_ROUND_UP(max_tokens_per_frame_, KALDI_CUDA_DECODER_1D_BLOCK) + 1);
		d_main_q_arc_offsets_.Resize(nchannels_,  max_tokens_per_frame_);
		d_hashmap_values_.Resize(nlanes_, hashmap_capacity_);
		frame_offsets_.resize(nchannels);

		d_main_q_acoustic_cost_.Resize(nlanes_, max_tokens_per_frame_);

		// We could remove this one
		d_aux_q_acoustic_cost_.Resize(nlanes_, max_tokens_per_frame_);

		// TODO use infotoken for next
		
		cudaMalloc(&d_extra_cost_concat_, sizeof(*d_extra_cost_concat_) * (size_t)max_tokens_per_frame_ * nlanes_); // FIXME cudafree, use main malloc

		d_acoustic_cost_concat_= d_aux_q_acoustic_cost_.lane(0);
		cudaMallocHost(&h_extra_cost_concat_, nlanes_*max_tokens_per_frame_*sizeof(*h_extra_cost_concat_));
		cudaMallocHost(&h_acoustic_cost_concat_, nlanes_*max_tokens_per_frame_*sizeof(*h_acoustic_cost_concat_));
		cudaMallocHost(&h_extra_prev_tokens_concat_, nlanes_*max_tokens_per_frame_*sizeof(*h_extra_prev_tokens_concat_));
		h_all_tokens_extra_prev_tokens_extra_cost_.resize(nchannels_);
		h_all_tokens_acoustic_cost_.resize(nchannels_);
		h_all_tokens_extra_prev_tokens_.resize(nchannels_);
		for(int32 ichannel=0; ichannel<nchannels_; ++ichannel) {
			h_all_tokens_extra_prev_tokens_extra_cost_[ichannel].reserve(max_tokens_);
			h_all_tokens_acoustic_cost_[ichannel].reserve(max_tokens_);
		}

		h_all_tokens_info_.resize(nchannels_);
		for(int32 ichannel=0; ichannel<nchannels_; ++ichannel) {
			h_all_tokens_info_[ichannel].reserve(max_tokens_);
		}
		d_infotoken_concat_ = d_aux_q_info_.lane(0);
		cudaMallocHost(&h_infotoken_concat_, nlanes_*max_tokens_per_frame_*sizeof(*h_infotoken_concat_));
		h_main_q_end_lane_offsets_.resize(nlanes_+1);
		h_emitting_main_q_end_lane_offsets_.resize(nlanes_+1);
		h_n_extra_prev_tokens_lane_offsets_.resize(nlanes_+1);

		// Init hashmap
/*
		cudaMalloc(&d_map_values_, nlanes_capacity_*sizeof(*d_map_values_));	
		init_hashmap_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(capacity, nlanes_),
			KALDI_CUDA_DECODER_1D_BLOCK,
			0,
			stream_>>>();
*/


		// Concat for copies 

		// Setting Kernel Params
		// sent to kernels by copy
		// Making sure we'll be able to send it to the kernels
		//KALDI_STATIC_ASSERT(sizeof(KernelParams) < KALDI_CUDA_DECODER_MAX_KERNEL_ARGUMENTS_BYTE_SIZE); TODO find include

		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemsetAsync(d_channels_counters_.MutableData(), 0, nchannels_*sizeof(*d_channels_counters_.MutableData())));
		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemsetAsync(d_lanes_counters_.MutableData(), 0, nlanes_*sizeof(*d_lanes_counters_.MutableData())));
		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMallocHost(&h_lanes_counters_, nlanes_ * sizeof(*h_lanes_counters_)));
		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMallocHost(&h_channels_counters_, nchannels_ * sizeof(*h_channels_counters_)));

		h_device_params_ = new DeviceParams();
		h_device_params_->d_channels_counters = d_channels_counters_.GetInterface();
		h_device_params_->d_lanes_counters =d_lanes_counters_.GetInterface();
		h_device_params_->d_main_q_state_and_cost = d_main_q_state_and_cost_.GetInterface();
		h_device_params_->d_main_q_info = d_main_q_info_.GetInterface();
		h_device_params_->d_aux_q_state_and_cost = d_aux_q_state_and_cost_.GetInterface();
		h_device_params_->d_main_q_extra_cost = d_main_q_extra_cost_.GetInterface();
		h_device_params_->d_main_q_acoustic_cost = d_main_q_acoustic_cost_.GetInterface();
		h_device_params_->d_aux_q_acoustic_cost = d_aux_q_acoustic_cost_.GetInterface();
		h_device_params_->d_aux_q_info = d_aux_q_info_.GetInterface();
		h_device_params_->d_main_q_degrees_prefix_sum = d_main_q_degrees_prefix_sum_.GetInterface();
		h_device_params_->d_main_q_block_sums_prefix_sum = d_main_q_block_sums_prefix_sum_.GetInterface();
		h_device_params_->d_main_q_representative_id = d_main_q_representative_id_.GetInterface();
		h_device_params_->d_main_q_extra_prev_tokens_prefix_sum = d_main_q_extra_prev_tokens_prefix_sum_.GetInterface();
		h_device_params_->d_main_q_n_extra_prev_tokens_local_idx = d_main_q_n_extra_prev_tokens_local_idx_.GetInterface();
		h_device_params_->d_main_q_extra_prev_tokens = d_main_q_extra_prev_tokens_.GetInterface();
		h_device_params_->d_main_q_arc_offsets = d_main_q_arc_offsets_.GetInterface();
		h_device_params_->d_hashmap_values = d_hashmap_values_.GetInterface();
		h_device_params_->d_histograms = d_histograms_.GetInterface();
		h_device_params_->d_arc_e_offsets = fst_.d_e_offsets_;
		h_device_params_->d_arc_ne_offsets = fst_.d_ne_offsets_;
		h_device_params_->d_arc_pdf_ilabels = fst_.d_arc_pdf_ilabels_;
		h_device_params_->d_arc_weights = fst_.d_arc_weights_;
		h_device_params_->d_arc_nextstates = fst_.d_arc_nextstates_;
		h_device_params_->d_fst_final_costs = fst_.d_final_;
		h_device_params_->default_beam = default_beam_;
		h_device_params_->lattice_beam = lattice_beam_;
		h_device_params_->generate_lattices = generate_lattices_;
		h_device_params_->q_capacity = max_tokens_per_frame_; 
		h_device_params_->init_channel_id = init_channel_id_; 
		h_device_params_->max_nlanes = nlanes_; 
		h_device_params_->nstates = fst_.num_states_; 
		h_device_params_->init_state = fst_.Start();
		KALDI_ASSERT(h_device_params_->init_state != fst::kNoStateId);
		h_device_params_->init_cost = StdWeight::One().Value();
		h_device_params_->hashmap_capacity = hashmap_capacity_;
		h_device_params_->max_active = max_active_;
		
		// For the first static_beam_q_length elements of the queue, we will keep the beam static
		int32 static_beam_q_length = max_tokens_per_frame_ / KALDI_CUDA_DECODER_ADAPTIVE_BEAM_STATIC_SEGMENT;
		// For the last adaptive_beam_q_length elements of the queue, we will decrease the beam, segment by segment
		int32 adaptive_beam_q_length = (max_tokens_per_frame_ - static_beam_q_length);
		int32 adaptive_beam_bin_width = adaptive_beam_q_length / KALDI_CUDA_DECODER_ADAPTIVE_BEAM_NBINS;
		h_device_params_->adaptive_beam_static_segment = static_beam_q_length; 
		h_device_params_->adaptive_beam_bin_width = adaptive_beam_bin_width; 

		// Reusing aux_q memory to list final states in GetLattice
		// Those cannot be used at the same time
		h_device_params_->d_list_final_tokens_in_main_q = d_aux_q_state_and_cost_.GetInterface();
		h_kernel_params_ = new KernelParams();

		init_hashmap_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(hashmap_capacity_, nlanes_),
			KALDI_CUDA_DECODER_1D_BLOCK,
			0,
			compute_st_>>>(*h_device_params_,*h_kernel_params_);
	  KALDI_DECODER_CUDA_CHECK_ERROR();


		ComputeInitialChannel();
		--nchannels_; // removing the init_channel_params from general list

		KALDI_DECODER_CUDA_CHECK_ERROR();
		num_frames_decoded_.resize(nchannels_, 0);

		// Making sure that everything is ready to use
		cudaStreamSynchronize(compute_st_);
	}

	CudaDecoder::~CudaDecoder() {
		cudaStreamDestroy(compute_st_);
		cudaStreamDestroy(copy_st_);

		cudaEventDestroy(can_read_h_main_q_narcs_);
		cudaEventDestroy(can_write_to_main_q_);
		cudaEventDestroy(can_read_final_h_main_q_end_);
		cudaEventDestroy(before_finalize_nonemitting_kernel_);
		cudaEventDestroy(can_use_extra_cost_);
		cudaEventDestroy(can_use_infotoken_);
		cudaEventDestroy(can_use_acoustic_cost_);

		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaFreeHost(h_lanes_counters_));
		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaFreeHost(h_channels_counters_));
		
		if(generate_lattices_) {
			cudaFreeHost(h_extra_cost_concat_);
			cudaFreeHost(h_acoustic_cost_concat_);
			cudaFreeHost(h_extra_prev_tokens_concat_);
		}
		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaFreeHost(h_infotoken_concat_));

		// Will call the cudaFrees inside destructors 
		delete h_kernel_params_;
		delete h_device_params_;

		KALDI_DECODER_CUDA_CHECK_ERROR();
	}

	void CudaDecoder::ComputeInitialChannel() {
		KALDI_ASSERT(nlanes_ > 0);
		const int32 ilane = 0;
		KALDI_ASSERT(ilane == 0);
		// Following kernels working channel_id
		h_kernel_params_->channel_to_compute[ilane] = init_channel_id_;
		h_kernel_params_->nlanes_used = 1;

		// Adding the start state to the initial token queue
		initialize_initial_lane_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, 1),
			KALDI_CUDA_DECODER_ONE_THREAD_BLOCK,
			0,
			compute_st_>>>(*h_device_params_);
    KALDI_DECODER_CUDA_CHECK_ERROR();

		// Initial ProcessNonEmitting
		preprocess_and_contract_kernel
			<<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, 1),
			KALDI_CUDA_DECODER_1D_BLOCK,
			0,
			compute_st_>>>(*h_device_params_,*h_kernel_params_);
    KALDI_DECODER_CUDA_CHECK_ERROR();

		finalize_process_non_emitting_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, 1),
			KALDI_CUDA_DECODER_LARGEST_1D_BLOCK,
			0,
			compute_st_>>>(*h_device_params_,*h_kernel_params_);
    KALDI_DECODER_CUDA_CHECK_ERROR();

		int32 main_q_end;
		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpyAsync(&main_q_end, 
				&d_lanes_counters_.lane(ilane)->main_q_narcs_and_end.y, 
				sizeof(int32), 
				cudaMemcpyDeviceToHost, 
				compute_st_));
    KALDI_DECODER_CUDA_CHECK_ERROR();
		cudaStreamSynchronize(compute_st_);
    KALDI_DECODER_CUDA_CHECK_ERROR();

		KALDI_ASSERT(main_q_end > 0);

		// Preparing for first frame + reverting back to init state (lookup table, etc.)
		fill_best_int_cost_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(main_q_end, 1),
			KALDI_CUDA_DECODER_1D_BLOCK,
			0,
			compute_st_>>>(*h_device_params_, *h_kernel_params_);
    KALDI_DECODER_CUDA_CHECK_ERROR();

		preprocess_in_place_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(main_q_end, 1),
			KALDI_CUDA_DECODER_1D_BLOCK,
			0,
			compute_st_>>>(*h_device_params_,*h_kernel_params_);
    KALDI_DECODER_CUDA_CHECK_ERROR();

		exclusive_sum_batched_step2_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(main_q_end, 1),
			KALDI_CUDA_DECODER_ONE_THREAD_BLOCK,
			0,
			compute_st_>>>(*h_device_params_,*h_kernel_params_);
    KALDI_DECODER_CUDA_CHECK_ERROR();

		exclusive_sum_batched_step3_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(main_q_end, 1),
			KALDI_CUDA_DECODER_1D_BLOCK,
			0,
			compute_st_>>>(*h_device_params_,*h_kernel_params_);
    KALDI_DECODER_CUDA_CHECK_ERROR();

		fill_extra_prev_tokens_list_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(main_q_end, 1),
			KALDI_CUDA_DECODER_1D_BLOCK,
			0,
			compute_st_>>>(*h_device_params_,*h_kernel_params_);
    KALDI_DECODER_CUDA_CHECK_ERROR();

		clear_hashmap_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(main_q_end, 1),
			KALDI_CUDA_DECODER_1D_BLOCK,
			0,
			compute_st_>>>(*h_device_params_,*h_kernel_params_);
    KALDI_DECODER_CUDA_CHECK_ERROR();
	
		// Saving init params on host
		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpyAsync(h_lanes_counters_, 
				d_lanes_counters_.MutableData(), 
				1*sizeof(*h_lanes_counters_), 
				cudaMemcpyDeviceToHost,
				compute_st_));

		// Waiting for compute to be done 
		cudaStreamSynchronize(compute_st_);
		KALDI_DECODER_CUDA_CHECK_ERROR();

		h_all_tokens_info_[init_channel_id_].resize(main_q_end);
		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpyAsync(&h_all_tokens_info_[init_channel_id_][0],
				d_main_q_info_.lane(ilane),
				main_q_end*sizeof(InfoToken),
				cudaMemcpyDeviceToHost,
				compute_st_));

		h_all_tokens_acoustic_cost_[init_channel_id_].resize(main_q_end);
		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpyAsync(&h_all_tokens_acoustic_cost_[init_channel_id_][0],
				d_main_q_acoustic_cost_.lane(ilane),
				main_q_end*sizeof(CostType),
				cudaMemcpyDeviceToHost,
				compute_st_));

		int32 main_q_n_extra_prev_tokens = h_lanes_counters_[ilane].main_q_n_extra_prev_tokens;
		h_all_tokens_extra_prev_tokens_[init_channel_id_].resize(main_q_n_extra_prev_tokens);
		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpyAsync(&h_all_tokens_extra_prev_tokens_[init_channel_id_][0],
				d_main_q_extra_prev_tokens_.lane(ilane),
				main_q_n_extra_prev_tokens*sizeof(*d_main_q_extra_prev_tokens_.lane(ilane)),
				cudaMemcpyDeviceToHost,
				compute_st_));
	
		h_all_tokens_extra_prev_tokens_extra_cost_[init_channel_id_].resize(main_q_n_extra_prev_tokens);
		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpyAsync(&h_all_tokens_extra_prev_tokens_extra_cost_[init_channel_id_][0],
				d_main_q_extra_cost_.lane(ilane),
				main_q_n_extra_prev_tokens*sizeof(*d_main_q_extra_cost_.lane(ilane)),
				cudaMemcpyDeviceToHost,
				compute_st_));

				
		// Context switch : saving channel state
		save_channels_state_from_lanes_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, 1),
						KALDI_CUDA_DECODER_ONE_THREAD_BLOCK,
						0,
						compute_st_>>>(*h_device_params_,*h_kernel_params_);
		KALDI_DECODER_CUDA_CHECK_ERROR();
	
		cudaStreamSynchronize(compute_st_);
		SaveChannelsStateFromLanesCPU();

		KALDI_DECODER_CUDA_CHECK_ERROR();
	}

	void CudaDecoder::InitDecoding() {
		std::vector<ChannelId> channels = {0};	
		InitDecoding(channels);
	}

	void CudaDecoder::InitDecoding(const std::vector<ChannelId> &channels) {
		const int nlanes_used = channels.size();
		// Getting *h_kernel_params ready to use
		SetChannelsInKernelParams(channels);
		KALDI_ASSERT(nlanes_used == h_kernel_params_->nlanes_used);

		// Size of the initial main_q_size
		const int32 init_main_q_size = h_channels_counters_[init_channel_id_].prev_main_q_narcs_and_end.y;

		KALDI_ASSERT(init_main_q_size > 0);
		// Getting the channels ready to compute new utterances
		init_decoding_on_device_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(init_main_q_size, nlanes_used),
						KALDI_CUDA_DECODER_1D_BLOCK,
						0,
						compute_st_>>>(*h_device_params_,*h_kernel_params_);
    KALDI_DECODER_CUDA_CHECK_ERROR();

		cudaStreamSynchronize(compute_st_);
		KALDI_DECODER_CUDA_CHECK_ERROR();
		for(ChannelId ichannel : channels) {
			// Tokens from initial main_q needed on host
			// Deep copy
			h_all_tokens_info_[ichannel] = h_all_tokens_info_[init_channel_id_];
			h_all_tokens_acoustic_cost_[ichannel] = h_all_tokens_acoustic_cost_[init_channel_id_];
			h_all_tokens_extra_prev_tokens_[ichannel] = h_all_tokens_extra_prev_tokens_[init_channel_id_];
			h_all_tokens_extra_prev_tokens_extra_cost_[ichannel] = h_all_tokens_extra_prev_tokens_extra_cost_[init_channel_id_];

			int32 n_initial_tokens = h_all_tokens_info_[init_channel_id_].size();

			h_channels_counters_[ichannel] = h_channels_counters_[init_channel_id_];
			num_frames_decoded_[ichannel] = 0;
			frame_offsets_[ichannel].clear();	
			frame_offsets_[ichannel].push_back(0);	
			frame_offsets_[ichannel].push_back(n_initial_tokens);	
		}
	}

	// Context-switch : Load and Store
	void CudaDecoder::LoadChannelsStateToLanesCPU() {
		for(LaneId ilane=0; ilane<h_kernel_params_->nlanes_used; ++ilane) {
			const ChannelId ichannel = h_kernel_params_->channel_to_compute[ilane];
			h_lanes_counters_[ilane].main_q_narcs_and_end = h_channels_counters_[ichannel].prev_main_q_narcs_and_end;
		}	
	}

	void CudaDecoder::SaveChannelsStateFromLanesCPU() {
		for(LaneId ilane=0; ilane<h_kernel_params_->nlanes_used; ++ilane) {
			const ChannelId ichannel = h_kernel_params_->channel_to_compute[ilane];
			h_channels_counters_[ichannel].prev_main_q_narcs_and_end = h_lanes_counters_[ilane].main_q_narcs_and_end;
			h_channels_counters_[ichannel].prev_main_q_global_offset = h_lanes_counters_[ilane].main_q_global_offset;
		}
	}
/*
	void CudaDecoder::AdvanceDecoding(DecodableInterface *decodable,
			int32 max_num_frames) {
		std::vector<ChannelId> channels = {0};	
		std::vector<DecodableInterface*> decodables = {decodable};	
		AdvanceDecoding(channels, decodables, max_num_frames);
	}
*/

	int32 CudaDecoder::GetMaxForAllLanes(std::function<int32(const LaneCounters &)> func) {
		int32 max_val = 0;
		const int32 nlanes_used = h_kernel_params_->nlanes_used;
		for(LaneId ilane = 0; ilane<nlanes_used; ++ilane) {
			const int32 val = func(h_lanes_counters_[ilane]);
			max_val = std::max(max_val, val); 
		}
		return max_val;
	}

	void CudaDecoder::CopyLaneCountersToHostAsync(cudaStream_t st) {
		const int32 nlanes_used = h_kernel_params_->nlanes_used;
		cudaMemcpyAsync(h_lanes_counters_,     
				d_lanes_counters_.MutableData(), 
				nlanes_used*sizeof(*h_lanes_counters_), 
				cudaMemcpyDeviceToHost,
				st);
	}

	template<typename T>
		void CudaDecoder::PerformConcatenatedCopy(std::function<int32(const LaneCounters &)> func,
				LaneMatrixInterface<T> src,
				T *d_concat,
				T *h_concat,
				cudaStream_t st,
				std::vector<int32> *lanes_offsets_ptr) {
			const int32 nlanes_used = h_kernel_params_->nlanes_used;

			int32 lane_offset = 0;
			int32 max_val = 0;
			std::vector<int32> &lanes_offsets = *lanes_offsets_ptr;
			KALDI_ASSERT(lanes_offsets.size() >= (nlanes_used+1));
			for(LaneId ilane=0; ilane<nlanes_used; ++ilane) {
				const int32 val = func(h_lanes_counters_[ilane]);
				max_val = std::max(max_val, val);
				lanes_offsets[ilane] = lane_offset;
				h_kernel_params_->main_q_end_lane_offsets[ilane] = lane_offset;
				lane_offset += val;
			}
			lanes_offsets[nlanes_used] = lane_offset;
			h_kernel_params_->main_q_end_lane_offsets[nlanes_used] = lane_offset;
			int32 sum_val = lane_offset;
			if(sum_val == 0)
				return; // nothing to do

			concatenate_lanes_data<T><<<KALDI_CUDA_DECODER_NUM_BLOCKS(max_val, nlanes_used),
				KALDI_CUDA_DECODER_1D_BLOCK,
				0,
				st>>>(*h_device_params_, 
			  		*h_kernel_params_, 
					src,	
					d_concat);
	  KALDI_DECODER_CUDA_CHECK_ERROR();

	  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpyAsync(h_concat,
					d_concat,
					sum_val* sizeof(T),
					cudaMemcpyDeviceToHost,
					st));

		}

	// One sync has to happen between PerformConcatenatedCopy and MoveConcatenatedCopyToVector
	template<typename T>
		void CudaDecoder::MoveConcatenatedCopyToVector(const std::vector<int32> &lanes_offsets,
				T *h_concat,
				std::vector<std::vector<T>> *vecvec) {
			const int32 nlanes_used = h_kernel_params_->nlanes_used;
			for(LaneId ilane=0; ilane<nlanes_used; ++ilane) {
				int32 beg = lanes_offsets[ilane];
				int32 end = lanes_offsets[ilane+1];
				ChannelId ichannel = h_kernel_params_->channel_to_compute[ilane];
				auto &vec = (*vecvec)[ichannel];
				vec.insert(vec.end(), h_concat+beg, h_concat+end);
			}
		}	

	void CudaDecoder::ApplyMaxActiveAndReduceBeam(bool use_aux_q) {
		auto func_aux_q_end = [] (const LaneCounters &c) { return c.post_expand_aux_q_end; };
		auto func_main_q_end = [] (const LaneCounters &c) { return c.main_q_narcs_and_end.y; };
		int32 max_q_end = use_aux_q
				? GetMaxForAllLanes(func_aux_q_end)
				: GetMaxForAllLanes(func_main_q_end);
		const int32 nlanes_used = h_kernel_params_->nlanes_used;
		
		// Adding a tolerance on max_active_
		// This is because we will usually not be able to limit the number of tokens
		// to exactly max_active
		// We will set it as close as possible to max_active, and we don't want to
		// keep calling the histograms kernels for a few tokens above the limit
		int32 thresh = (int32)(1.2*max_active_); // TODO constant
		if(max_q_end <= thresh) { 
			// The queues are already smaller than max_active_
			// nothing to do
			return;
		}
		
		compute_costs_histogram_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(max_q_end, nlanes_used),
			KALDI_CUDA_DECODER_1D_BLOCK,
			0,
			compute_st_>>>(*h_device_params_,*h_kernel_params_, use_aux_q);
	  KALDI_DECODER_CUDA_CHECK_ERROR();

		update_beam_using_histogram_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, nlanes_used),
			KALDI_CUDA_DECODER_1D_BLOCK,
			0,
			compute_st_>>>(*h_device_params_,*h_kernel_params_, use_aux_q); 
	  KALDI_DECODER_CUDA_CHECK_ERROR();
	}

	void CudaDecoder::AdvanceDecoding(const std::vector<ChannelId> &channels,
			std::vector<CudaDecodableInterface*> &decodables,
			int32 max_num_frames) {
		const int nlanes_used = channels.size();
		if(nlanes_used <= 0)
			return;
		// How many frames should we decode ?
		int32 nframes_to_decode = INT_MAX;
		//std::vector<int> debug_ntokens;
		//std::vector<int> debug_narcs;
		for(int32 ilane=0; ilane<nlanes_used; ++ilane) {
			const ChannelId ichannel = channels[ilane];
			const int32 num_frames_decoded = num_frames_decoded_[ichannel];
			KALDI_ASSERT(num_frames_decoded >= 0 &&
					"You must call InitDecoding() before AdvanceDecoding()");
			int32 num_frames_ready = decodables[ilane]->NumFramesReady(); // FIXME plug the right one
			// num_frames_ready must be >= num_frames_decoded, or else
			// the number of frames ready must have decreased (which doesn't
			// make sense) or the decodable object changed between calls
			// (which isn't allowed).
			KALDI_ASSERT(num_frames_ready >= num_frames_decoded);
			int32 channel_nframes_to_decode = num_frames_ready - num_frames_decoded;
			nframes_to_decode = std::min(nframes_to_decode,
						channel_nframes_to_decode);
		}	
		if(max_num_frames >= 0)
			nframes_to_decode = std::min(nframes_to_decode, max_num_frames);

		// Getting *h_kernel_params ready to use
		SetChannelsInKernelParams(channels);
		KALDI_ASSERT(nlanes_used == h_kernel_params_->nlanes_used);

		// Getting the lanes ready to work with those channels  
		load_channels_state_in_lanes_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, nlanes_used),
						KALDI_CUDA_DECODER_ONE_THREAD_BLOCK,
						0,
						compute_st_>>>(*h_device_params_,*h_kernel_params_);
    KALDI_DECODER_CUDA_CHECK_ERROR();

		LoadChannelsStateToLanesCPU();
		nvtxRangePushA("Decoding");
		std::vector<int32> main_q_emitting_end(nlanes_used);
		for(int32 iframe=0; iframe<nframes_to_decode; ++iframe)  {
			//int32 debug_f_narcs = 0;
			// Computing a new frame
			// Loglikelihoods from the acoustic model
			//if(iframe > 2)  KALDI_ASSERT(0);
			nvtxRangePop(); // Decoding
			for(LaneId ilane=0; ilane<h_kernel_params_->nlanes_used; ++ilane) {
				ChannelId ichannel = h_kernel_params_->channel_to_compute[ilane];
				int32 frame = num_frames_decoded_[ichannel];
				h_kernel_params_->loglikelihoods_ptrs[ilane] = decodables[ilane]->GetLogLikelihoodsCudaPointer(frame);
			}
  			cudaStreamSynchronize(cudaStreamPerThread);      // Nnet3 sync TODO do a GetNnet3CudaStream
			nvtxRangePushA("Decoding");

			// ProcessEmitting 
			// 
			// Before executing ProcessEmitting, we have :
			// - The main_q contains tokens from the last frame
			// - The aux_q is empty
			//
			// ProcessEmitting will do the operation :
			//
			// read tokens from main_q ----FST---> create new tokens in the aux_q
			//
			// We will not write in the main q in that step
			// The input tokens are already in the main_q
			// (they were put there by the ProcessNonemittings 
			// from the previous frame)
			// We don't need can_write_to_main_q_
			// because we won't write to the main_q
			// The output tokens will go to aux_q

			// ProcessEmitting generates tokens associated with the new frame i
			// When we call ProcessEmitting, the main_q contains the tokens associated
			// with the previous frame (i-1). Using d_main_q_state and the emitting arcs from the FST graph,
			// we create a new tokens queue, which will be stored in the aux_q

			// Process emitting, expanding arcs
			// Looking for the channel with max numbers of arcs
			{
				auto func_narcs = [] (const LaneCounters &c) { return c.main_q_narcs_and_end.x; };
				int32 max_main_q_narcs = GetMaxForAllLanes(func_narcs);
				KALDI_ASSERT(max_main_q_narcs > 0);

				expand_arcs_kernel<true><<<KALDI_CUDA_DECODER_NUM_BLOCKS(max_main_q_narcs, nlanes_used),
					KALDI_CUDA_DECODER_1D_BLOCK,
					0,
					compute_st_>>>(*h_device_params_,*h_kernel_params_);
        KALDI_DECODER_CUDA_CHECK_ERROR();

				// Updating a few counters, like resetting aux_q_end to 0...
				// true is for IS_EMITTING
				post_expand_kernel<true><<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, nlanes_used),
					KALDI_CUDA_DECODER_ONE_THREAD_BLOCK,
					0,
					compute_st_>>>(*h_device_params_,*h_kernel_params_);
        KALDI_DECODER_CUDA_CHECK_ERROR();
			}
			// After ProcessEmitting we won't need the token
			// associated with the previous frame anymore
			// At the end of ProcessEmitting the main_q was flushed 
			// (by setting main_q_end == 0)
			// Tokens that were flushed at that step have been previously 
			// moved to the host memory 
			// We update the global offset of the main_q
			// the global offset takes into account all tokens that have been moved
			// to the host memory

			// ProcessNonemitting
			//
			// Processing non emitting arcs
			//
			// The operation is :
			//
			// PreprocessAndContract:
			// read input tokens from aux_q 
			//     ---contract (prune)--->
			// write non-pruned input tokens to main_q (append at the end of the queue)
			//
			// ExpandArc:
			// read input tokens from main_q 
			//     ---FST--->
			// create new tokens in the aux_q
			//
			// We then iterate those operations until no new tokens are created 
			//

			// We will write to main_q. We need it to be ready
			// for next kernels on compute_st_ 
			bool first_nonemitting = true;;
			while(true) {
				// Moving the lanes_params to host,
				// to have the aux_q_end values
				CopyLaneCountersToHostAsync(compute_st_);
				cudaStreamSynchronize(compute_st_);
        CheckOverflow();
				{
					// If one of the aux_q contains more than max_active_ tokens,
					// we'll reduce the beam to only keep max_active_ tokens
					ApplyMaxActiveAndReduceBeam(true);

					auto func_aux_q_end = [] (const LaneCounters &c) { return c.post_expand_aux_q_end; };
					int32 max_aux_q_end = GetMaxForAllLanes(func_aux_q_end);

					// aux_q_end == 0, not likely, but possible
					if(max_aux_q_end == 0) 
						break; 

					preprocess_and_contract_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(max_aux_q_end, nlanes_used),
						KALDI_CUDA_DECODER_1D_BLOCK,
						0,
						compute_st_>>>(*h_device_params_,*h_kernel_params_);
          KALDI_DECODER_CUDA_CHECK_ERROR();
				}
				// We need main_q_narcs and main_q_end after contract
				CopyLaneCountersToHostAsync(compute_st_);
				cudaStreamSynchronize(compute_st_);
        CheckOverflow();
				
        // We'll need max_main_q_narcs for the next expand
				// We also need to copy the acoustic costs back to host.
				// We'll concatenate the costs from the different lanes into in a single
				// continuous array.
				{
					if(first_nonemitting) {
						auto func_main_q_end = [] (const LaneCounters &c) { return c.main_q_narcs_and_end.y; };
						PerformConcatenatedCopy(func_main_q_end,
								h_device_params_->d_main_q_acoustic_cost,
								d_acoustic_cost_concat_,
								h_acoustic_cost_concat_,
								compute_st_,
								&h_emitting_main_q_end_lane_offsets_);
						for(int32 ilane=0; ilane < nlanes_used; ++ilane)
							main_q_emitting_end[ilane] = h_lanes_counters_[ilane].main_q_narcs_and_end.y;
						first_nonemitting = false;
					}
				}
				{
					// For next expand
					auto func_main_q_narcs = [] (const LaneCounters &c) { return c.main_q_narcs_and_end.x; };
					int32 max_main_q_narcs = GetMaxForAllLanes(func_main_q_narcs);

					// If == 0, we will never break out of that while(true) loop
					KALDI_ASSERT(KALDI_CUDA_DECODER_NONEM_LT_MAX_NARCS > 0); 
					// If we have only a few arcs, jumping to the one-CTA per lane persistent version
					if(max_main_q_narcs < KALDI_CUDA_DECODER_NONEM_LT_MAX_NARCS) {
						break;
					}

					// false is for non emitting
					expand_arcs_kernel<false><<<KALDI_CUDA_DECODER_NUM_BLOCKS(max_main_q_narcs, nlanes_used),
						KALDI_CUDA_DECODER_1D_BLOCK,
						0,
						compute_st_>>>(*h_device_params_,*h_kernel_params_);
          KALDI_DECODER_CUDA_CHECK_ERROR();

					// false is for non emitting
					post_expand_kernel<false><<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, nlanes_used),
						KALDI_CUDA_DECODER_ONE_THREAD_BLOCK,
						0,
						compute_st_>>>(*h_device_params_,*h_kernel_params_);
          KALDI_DECODER_CUDA_CHECK_ERROR();
				}
			}
			// Finalizing process non emitting. Takes care of the long tail, 
			// the final iterations with a small numbers of arcs. Do the work inside a single CTA (per lane),
			// using local __syncthreads() 
			finalize_process_non_emitting_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, nlanes_used),
							KALDI_CUDA_DECODER_LARGEST_1D_BLOCK,
							0,
							compute_st_>>>(*h_device_params_,*h_kernel_params_);
		  KALDI_DECODER_CUDA_CHECK_ERROR();

			// Moving back to host the final (for this frame) values of :
			// - main_q_end
			// - main_q_narcs
			CopyLaneCountersToHostAsync(compute_st_);
			// Waiting for the copy
			cudaStreamSynchronize(compute_st_);

      CheckOverflow();

			MoveConcatenatedCopyToVector(h_emitting_main_q_end_lane_offsets_,
					h_acoustic_cost_concat_,
					&h_all_tokens_acoustic_cost_);

			{
				// Post processing the tokens for that frame
				// - do the preprocess necessary for the next emitting expand (will happen with next frame)
				// - if a state S has more than one token associated to it, generate the list of those tokens
				// It allows to backtrack efficiently in GetRawLattice
				// - compute the extra costs
				auto func_main_q_end = [] (const LaneCounters &c) { return c.main_q_narcs_and_end.y; };
				int32 max_main_q_end = GetMaxForAllLanes(func_main_q_end);

				ApplyMaxActiveAndReduceBeam(false);
			
				fill_best_int_cost_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(max_main_q_end, nlanes_used),
					KALDI_CUDA_DECODER_1D_BLOCK,
					0,
					compute_st_>>>(*h_device_params_, *h_kernel_params_);
	      KALDI_DECODER_CUDA_CHECK_ERROR();

				preprocess_in_place_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(max_main_q_end, nlanes_used),
					KALDI_CUDA_DECODER_1D_BLOCK,
					0,
					compute_st_>>>(*h_device_params_,*h_kernel_params_);
	      KALDI_DECODER_CUDA_CHECK_ERROR();

				exclusive_sum_batched_step2_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, nlanes_used),
					KALDI_CUDA_DECODER_1D_BLOCK,
					0,
					compute_st_>>>(*h_device_params_,*h_kernel_params_);
	      KALDI_DECODER_CUDA_CHECK_ERROR();

				exclusive_sum_batched_step3_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(max_main_q_end, nlanes_used),
					KALDI_CUDA_DECODER_1D_BLOCK,
					0,
					compute_st_>>>(*h_device_params_,*h_kernel_params_);
	      KALDI_DECODER_CUDA_CHECK_ERROR();

				fill_extra_prev_tokens_list_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(max_main_q_end, nlanes_used),
					KALDI_CUDA_DECODER_1D_BLOCK,
					0,
					compute_st_>>>(*h_device_params_,*h_kernel_params_);
	      KALDI_DECODER_CUDA_CHECK_ERROR();

				clear_hashmap_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(max_main_q_end, nlanes_used),
					KALDI_CUDA_DECODER_1D_BLOCK,
					0,
					compute_st_>>>(*h_device_params_,*h_kernel_params_);
	      KALDI_DECODER_CUDA_CHECK_ERROR();

				// We need the main_q_narcs from preprocess_in_place
				CopyLaneCountersToHostAsync(compute_st_);
			}

			// Copying InfoTokens back to host
			{
				auto func_main_q_end = [] (const LaneCounters &c) { return c.main_q_narcs_and_end.y; };
				PerformConcatenatedCopy(func_main_q_end,
						h_device_params_->d_main_q_info,
						d_infotoken_concat_,
						h_infotoken_concat_,
						compute_st_,
						&h_main_q_end_lane_offsets_);
			}
		
			// Sync for :
			// - h_infotoken_concat_ copy done
			// - using lane_counters.main_q_n_extra_prev_tokens
			cudaStreamSynchronize(compute_st_);
      CheckOverflow();
		
			// Starting the extra_prev_tokens copies
			{
				auto func_main_q_n_extra_prev_tokens = [] (const LaneCounters &c) { return c.main_q_n_extra_prev_tokens; };
				PerformConcatenatedCopy(func_main_q_n_extra_prev_tokens,
						h_device_params_->d_main_q_extra_prev_tokens,
						d_infotoken_concat_, // FIXME use dedicated ptr
						h_extra_prev_tokens_concat_,
						compute_st_,
						&h_n_extra_prev_tokens_lane_offsets_);
				PerformConcatenatedCopy(func_main_q_n_extra_prev_tokens,
						h_device_params_->d_main_q_extra_cost,
						d_extra_cost_concat_, 
						h_extra_cost_concat_,
						compute_st_,
						&h_n_extra_prev_tokens_lane_offsets_);
			}
			
			// Moving infotokens to vecs
			MoveConcatenatedCopyToVector(h_main_q_end_lane_offsets_,
					h_infotoken_concat_,
					&h_all_tokens_info_);

			// Waiting for the copies
			cudaStreamSynchronize(compute_st_);

			// Moving the extra_prev_tokens to vecs	
			MoveConcatenatedCopyToVector(h_n_extra_prev_tokens_lane_offsets_,
					h_extra_prev_tokens_concat_,
					&h_all_tokens_extra_prev_tokens_);
			MoveConcatenatedCopyToVector(h_n_extra_prev_tokens_lane_offsets_,
					h_extra_cost_concat_,
					&h_all_tokens_extra_prev_tokens_extra_cost_);

			// Adding 0.0f acoustic_costs for non-emittings 
			for(LaneId ilane=0; ilane<nlanes_used; ++ilane) {
				const ChannelId ichannel = h_kernel_params_->channel_to_compute[ilane];
				++num_frames_decoded_[ichannel];
				const int32 main_q_end = h_lanes_counters_[ilane].main_q_narcs_and_end.y;
				frame_offsets_[ichannel].push_back(frame_offsets_[ichannel].back() + main_q_end);
				int32 ntokens_nonemitting = main_q_end - main_q_emitting_end[ilane];
				auto &vec = h_all_tokens_acoustic_cost_[ichannel];
				vec.insert(vec.end(), ntokens_nonemitting, 0.0f);
				KALDI_ASSERT(vec.size() == h_all_tokens_info_[ichannel].size());
				KALDI_ASSERT(h_all_tokens_extra_prev_tokens_[ichannel].size() == h_all_tokens_extra_prev_tokens_[ichannel].size());
			}	

			CheckOverflow();
			KALDI_DECODER_CUDA_CHECK_ERROR();
		}   

		// Context switch : saving channels states
		save_channels_state_from_lanes_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, nlanes_used),
			KALDI_CUDA_DECODER_ONE_THREAD_BLOCK,
			0,
			compute_st_>>>(*h_device_params_,*h_kernel_params_);
		KALDI_DECODER_CUDA_CHECK_ERROR();
		SaveChannelsStateFromLanesCPU();

		/*	
		int32 sum = std::accumulate(debug_ntokens.begin(), debug_ntokens.end(), 0);
		int32 arcs_sum = std::accumulate(debug_narcs.begin(), debug_narcs.end(), 0);
		double avg = ((double)sum)/(debug_ntokens.size());
		double narcs_avg = ((double)arcs_sum)/(debug_ntokens.size())/nlanes_used;
		printf("sum=%i, avg ntokens=%f, avg arcs=%f \n", sum, avg, narcs_avg);
		*/
		nvtxRangePop();
	}

	void CudaDecoder::CheckOverflow() {
		for(LaneId ilane=0; ilane<h_kernel_params_->nlanes_used; ++ilane) {
			bool q_overflow = h_lanes_counters_[ilane].q_overflow;
			if(q_overflow) {
        //TODO temporary until overflow handling is fixed
        throw CudaDecoderException("Overflow increase --max-tokens-per-frame", __FILE__, __LINE__, true);
				// An overflow was prevented in a kernel
				// The algorithm can still go on but quality of the result can be reduced
				// (less tokens were generated)
				KALDI_WARN << "Preventing overflow of the frame tokens. Pursuing "
					<< "execution but the quality of the output may be decreased. "
					<< "To prevent this from happening, please increase the parameter --max-tokens-per-frame"
					<< " and/or decrease --beam";
			}
		}
	}


	// GetBestCost
	// returns the minimum cost among all tokens cost in the current frame
	// also returns the index of one token with that min cost
	//
	// Only called at the end of the computation of one audio file
	// not optimized
	void CudaDecoder::GetBestCost(const std::vector<ChannelId> &channels, 
				bool use_final_costs, 
				std::vector<std::pair<int32,CostType>> *argmins, 
				std::vector<std::vector<std::pair<int,float>>> *list_finals_token_idx_and_cost, 
				std::vector<bool> *has_reached_final) {
		const int nlanes_used = channels.size();
		if(nlanes_used <= 0)
			return;

		list_finals_token_idx_and_cost->resize(nlanes_used);
		// Getting *h_kernel_params ready to use
		SetChannelsInKernelParams(channels);
		KALDI_ASSERT(nlanes_used == h_kernel_params_->nlanes_used);
		int32 max_main_q_end = 0;
		for(ChannelId ichannel : channels)
			max_main_q_end = std::max(max_main_q_end, h_channels_counters_[ichannel].prev_main_q_narcs_and_end.y); 

		// We already know what's the best cost, because we needed it for the cutoff
		// it was saved in channel_counters.prev_min_cost
		// we just need to find its index	
		get_best_cost_kernel_step1<<<KALDI_CUDA_DECODER_NUM_BLOCKS(max_main_q_end, nlanes_used),
				KALDI_CUDA_DECODER_1D_BLOCK,
				0,
				compute_st_>>>(*h_device_params_,*h_kernel_params_, use_final_costs, StdWeight::Zero().Value());
	  KALDI_DECODER_CUDA_CHECK_ERROR();

		get_best_cost_kernel_step2<<<KALDI_CUDA_DECODER_NUM_BLOCKS(max_main_q_end, nlanes_used),
				KALDI_CUDA_DECODER_1D_BLOCK,
				0,
				compute_st_>>>(*h_device_params_,*h_kernel_params_, use_final_costs, StdWeight::Zero().Value());
	
		KALDI_DECODER_CUDA_CHECK_ERROR();
		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpyAsync(h_lanes_counters_,     
				d_lanes_counters_.MutableData(), 
				nlanes_used*sizeof(*h_lanes_counters_), 
				cudaMemcpyDeviceToHost,
				compute_st_));

		argmins->clear();
		has_reached_final->clear();
		cudaStreamSynchronize(compute_st_);
		std::vector<int2> int2_buffer;
		for(int32 ilane=0; ilane<nlanes_used; ++ilane) {
			int2 minarg = h_lanes_counters_[ilane].min_int_cost_and_arg;
			CostType min_cost = orderedIntToFloatHost(minarg.x);
			int32 arg = minarg.y;
			argmins->push_back({arg,min_cost});
			int nfinals = h_lanes_counters_[ilane].nfinals;
			has_reached_final->push_back(h_lanes_counters_[ilane].has_reached_final);
			(*list_finals_token_idx_and_cost)[ilane].resize(nfinals);	
			int2_buffer.resize(nfinals);	
			cudaMemcpyAsync(&int2_buffer[0], 
				        d_aux_q_state_and_cost_.lane(ilane),
					nfinals*sizeof(int2),
					cudaMemcpyDeviceToHost,
					compute_st_);

			for(int i=0; i<nfinals; ++i) {
				int global_idx = int2_buffer[i].x;
				float cost_with_final = orderedIntToFloatHost(int2_buffer[i].y);
				//printf("cost with final = %f \n", cost_with_final);
				(*list_finals_token_idx_and_cost)[ilane][i].first = global_idx;
				(*list_finals_token_idx_and_cost)[ilane][i].second = cost_with_final;
				//printf("final on host = %i , %f \n", global_idx, cost_with_final);
			}
		}
		cudaStreamSynchronize(compute_st_);
	}

	//
	// GetBestPath is called at the end of the computation
	// It chooses the best token from the last frame, 
	// and backtracks all the path to the beginning (StartState)
	// from there
	// It then returns that path
	//
	bool CudaDecoder::GetBestPath(Lattice* fst_out, bool use_final_probs) {
		std::vector<ChannelId> channels = {0};	
		std::vector<Lattice*> fst_out_vec = {fst_out};	
		return GetBestPath(channels, fst_out_vec, use_final_probs); 
	}
	bool CudaDecoder::GetBestPath(const std::vector<ChannelId> &channels, std::vector<Lattice*> &fst_out_vec, bool use_final_probs) {
		KALDI_ASSERT(channels.size() == fst_out_vec.size());
		KALDI_ASSERT(channels.size() <= nchannels_);
		nvtxRangePushA("GetBestPath");
		std::vector<std::pair<int32,CostType>> argmins;
		std::vector<bool> has_reached_final;
		std::vector<std::vector<std::pair<int,float>>> list_finals_token_idx_and_cost;
		GetBestCost(channels, use_final_probs, &argmins, &list_finals_token_idx_and_cost, &has_reached_final);
		// TODO handle if a final state was not found

		// We want the copy to host of the last tokens to be done
		// we're going to read h_all_tokens_info
		cudaEventSynchronize(can_write_to_main_q_);
		for(int32 i=0; i<channels.size(); ++i) {
			const ChannelId ichannel = channels[i];
			const int32 token_with_best_cost = argmins[i].first;
			const bool isfinal = has_reached_final[i];
			int32 token_idx = token_with_best_cost;

			// Backtracking
			// Going all the way from the token with best cost
			// to the beginning (StartState)
			std::vector<int32> reversed_path;

			// The first token was inserted at the beginning of the queue
			// it always has index 0
			// We backtrack until that first token
			while(token_idx != 0) {
				InfoToken token = h_all_tokens_info_[ichannel][token_idx];
				// We want an arc with extra_cost == 0 
				int32 arc_idx, prev_token_idx;
				if(token.IsUniqueTokenForStateAndFrame()) {
					// If we have only one, it is an arc with extra_cost == 0
					arc_idx = token.arc_idx;
					prev_token_idx = token.prev_token;
				} else {
					// Using the first arc with extra_cost == 0
					int32 offset, size;
					std::tie(offset,size) = token.GetNextStateTokensList();
					bool found = false;
					for(auto i=0; i<size; ++i) {
						CostType arc_extra_cost = h_all_tokens_extra_prev_tokens_extra_cost_[ichannel][offset+i].x;
						if(arc_extra_cost == 0.0f) {
							InfoToken list_token = h_all_tokens_extra_prev_tokens_[ichannel][offset+i];
							arc_idx = list_token.arc_idx;
							prev_token_idx = list_token.prev_token;
							found = true;
							break;	
						}
					} 
					KALDI_ASSERT(found);
				}
				reversed_path.push_back(arc_idx);
				token_idx = prev_token_idx; 
			}

			Lattice *fst_out = fst_out_vec[i];

			// Reset the fst_out
			fst_out->DeleteStates();

			// Building the output Lattice
			StateId cur_state = fst_out->AddState();
			fst_out->SetStart(cur_state);

			for (int32 i=reversed_path.size()-1; i>=1; i--) {
				int32 arc_idx = reversed_path[i];

				LatticeArc arc(fst_.h_arc_id_ilabels_[arc_idx], 
						fst_.h_arc_olabels_[arc_idx],
						LatticeWeight(fst_.h_arc_weights_[arc_idx], 0), 
						fst_.h_arc_nextstates_[arc_idx]);

				arc.nextstate = fst_out->AddState();
				fst_out->AddArc(cur_state, arc);
				cur_state = arc.nextstate;
			}

			// Adding final cost to final state
			if (isfinal && use_final_probs)
				fst_out->SetFinal(cur_state,
						LatticeWeight(fst_.h_final_[fst_.h_arc_nextstates_[reversed_path[0]]], 0.0));
			else
				fst_out->SetFinal(cur_state, LatticeWeight::One());

			fst::RemoveEpsLocal(fst_out);


		}
		nvtxRangePop();
		return true;
	}

	bool CudaDecoder::GetRawLattice(const std::vector<ChannelId> &channels, std::vector<Lattice*> &fst_out_vec, bool use_final_probs) {
		KALDI_ASSERT(channels.size() == fst_out_vec.size());
		KALDI_ASSERT(channels.size() <= nchannels_);
		std::vector<std::pair<int32,CostType>> argmins;
		std::vector<bool> has_reached_final;
		std::vector<std::vector<std::pair<int,float>>> list_finals_token_idx_and_cost;
		GetBestCost(channels, use_final_probs, &argmins, &list_finals_token_idx_and_cost, &has_reached_final);

		// In some cases we can update an extra_cost that has already been used
		// For instance we process arcs in that order :
		// 1) a -> b, which updates extra_cost[b] using extra_cost[a]
		// 2) c -> a, which updates extra-cost[a] (using extra_cost[c])
		// because the arcs were not considered in topological order, we need to run again the step 1,
		// to get the correct extra_cost[b] (using the latest extra_cost[a])
		// However, we only re-run the step 1 if the value extra_cost[a] has changed for more than min_delta
		const float min_delta = 1e-5;
		for(int32 i=0; i<channels.size(); ++i) {
			nvtxRangePushA("GetRawLatticeOneChannel");
			const ChannelId ichannel = channels[i];
			const int32 nframes = NumFramesDecoded(ichannel);

			// Total number of tokens generated by the utterance on channel ichannel
			const int32 total_ntokens = h_all_tokens_info_[ichannel].size();

			// Returns a unique id for each (iframe, fst_state) pair
			// we simply use the fact that token_idx < total_ntokens
			// We need to be able to quickly identity a (iframe, fst_state) ID, without using the 
			// fst_state value, because we don't have it directly. In theory we could load it (using the arc_idx),
			// but it leads to a lot of cache misses, slowing everything down
			auto get_unique_id = [total_ntokens] (int32 token_idx, InfoToken token) {
				// If we have a unique token for this (frame,fst_state)
				// Then its ID is a unique ID for (frame,fst_state)
				if(token.IsUniqueTokenForStateAndFrame())
					return token_idx; 

				// If we have multiple tokens for this (frame,fst_state),
				// let's use the "extra_prev_tokens" offset, which is unique for (frame,fst_state) in that case
				
				// Adding the total_ntokens offset to avoid collisions with the previous case 
				return (total_ntokens + token.prev_token);
			};

			// Preparing output lattice
			// The start state has to be 0 (cf some asserts somewhere else in Kaldi)
			// Adding it now
			Lattice *fst_out = fst_out_vec[i];
			fst_out->DeleteStates();
			StateId fst_lattice_start = fst_out->AddState();
			fst_out->SetStart(fst_lattice_start);

			// Keeping track of a variety of info about states in the lattice
			// - token_extra_cost. A path going from the current lattice_state to the end has an extra cost
			// compared to the best path (which has an extra cost of 0). 
			// token_extra_cost is the minimum of the extra_cost of all paths going from the current lattice_state
			// to the final frame. 
			// - fst_lattice_state is the StateId of the lattice_state in fst_out (in the output lattice)
			// - is_state_closed is true if the token_extra_cost has been read by another token. It means that the
			// token_extra_cost value has been used, and if we modify token_extra_cost again, we may need to recompute things 
			struct RawLatticeState {
				CostType token_extra_cost;
				StateId fst_lattice_state;
				bool is_state_closed;
			};
			
			// Using one map per frame. We always know to which frame a token belongs. Using one big map slows everything down
			std::unordered_map<int32,RawLatticeState> prev_f_raw_lattice_state, curr_f_raw_lattice_state; 
			// We want the unicity of each arc_idx for one frame. Important because we can replay a frame (and possibly add multiple time the same arc)
			std::unordered_set<int32> f_arc_idx_added;
			// Keeping track of which tokens need to be computed. Think of those as FIFO queues,
			// except that we don't want to pop the front right away, because we may replay a frame
			// (and we need to remember what's in that frame)
			// We are also not using an iterator through the [prev|curr]_f_raw_lattice_state because we are 
			// sometime adding stuff in q_curr_frame_todo while reading it. If using a map we can possibly add the new
			// element before the current iterator 
			std::vector<std::pair<int32,InfoToken>> q_curr_frame_todo;
			std::vector<std::pair<int32,InfoToken>> q_prev_frame_todo;

			int32 nclosed = 0;

			// Reading the overall best_cost for that utterance's last frame. Was set by GetBestCost (cf above)
			const CostType best_cost = argmins[i].second;
			// Iterating through tokens associated with a final state in the last frame
			for(auto& p : list_finals_token_idx_and_cost[i]) {
				// This final token has a final cost of final_token_cost
				CostType final_token_cost = p.second;
				// This token has possibly an extra cost compared to the best
				CostType extra_cost = final_token_cost - best_cost;
				// We only want to keep paths that have a cost within [best; best+lattice_beam]
				if(extra_cost > lattice_beam_) {
					continue;
				}

				const int32 final_token_idx = p.first;
				InfoToken final_token = h_all_tokens_info_[ichannel][final_token_idx];
				
				// Unique ID for our (iframe, fst_state) pair 
				int32 lattice_state = get_unique_id(final_token_idx, final_token);
				decltype(curr_f_raw_lattice_state.end()) map_it;
				bool inserted;
				// We need to create the lattice_state linked to (iframe, state) in the lattice if it doesn't already exists
				// Inserts only if the key doesn't exist in the map
				std::tie(map_it, inserted) = curr_f_raw_lattice_state.insert({lattice_state, {FLT_MAX,-1,false}});
				// If we've inserted the element, it means that that state didn't exist in the map
				// Because this is a final state, we need to do a bit of extra work to add the final_cost to it

				if(inserted) {
					// We want to figure out which FST state this token is associated to
					// We don't have that info anymore, it wasn't transfered from the GPU
					// We still need it for final tokens, because we need to know which final cost to add
					// in the lattice. To find that state, we need the id of an arc going to that state,
					// then we'll look in the graph and figure out next_state[arc_idx]
					// we just need a valid arc_idx
					int32 arc_idx;
					if(final_token.IsUniqueTokenForStateAndFrame()) {
						// If unique, we can directly use this arc_idx
						arc_idx = final_token.arc_idx;
					} else {
						// If we have multiple tokens associated to that fst state, just pick the first one
						// from the list
						int32 offset, size;
						std::tie(offset,size) = final_token.GetNextStateTokensList();
						InfoToken prev_token = h_all_tokens_extra_prev_tokens_[ichannel][offset];
						arc_idx = prev_token.arc_idx;
					}
					// Creating the state associated with (iframe, fst_state) in the lattice
					StateId fst_lattice_final_state = fst_out->AddState();
					map_it->second.fst_lattice_state = fst_lattice_final_state; 
					q_curr_frame_todo.push_back({final_token_idx,final_token});
		
					if(has_reached_final[i]) {
						// If we have reached final states, adding the final cost
						// We now have a valid arc_idx. We can read the FST state
						StateId fst_next_state = fst_.h_arc_nextstates_[arc_idx];

						fst_out->SetFinal(fst_lattice_final_state,
								LatticeWeight(fst_.h_final_[fst_next_state], 0.0));
					} else {
						fst_out->SetFinal(fst_lattice_final_state,
								LatticeWeight::One());
					}
				}

				map_it->second.token_extra_cost = std::min(map_it->second.token_extra_cost, extra_cost);
			}

			// We're now going to backtrack frame by frame
			// For each frame we're going to process tokens that need to be inserted into the output lattice
			// and add their predecessors to the todo list
			for(int32 iframe=nframes; iframe >= 0; --iframe) {
				// Tokens for the current frame were inserted after this offset in the token list
				const int32 curr_frame_offset = frame_offsets_[ichannel][iframe];	

				// Tokens by themselves are in topological order. However, when creating the 
				// lattice, we merge tokens sharing the same (iframe, fst_state) pair. 
				// when merging those tokens, we break the topological order
				// and we may have to replay that frame. 
				bool must_replay_frame;
				do {
					must_replay_frame = false;
					// Reading something to do. We are pushing stuff back in q_curr_frame_todo while reading it,
					// so it's important to always read q_curr_frame_todo.size() directly
					for(int32 u=0; u<q_curr_frame_todo.size(); ++u) {
						int32 token_idx;
						InfoToken token;
						std::tie(token_idx,token) = q_curr_frame_todo[u];
						// Making sure the token is in the current frame
						KALDI_ASSERT(token_idx >= curr_frame_offset);
						StateId lattice_next_state = get_unique_id(token_idx, token);

						auto to_map_it = curr_f_raw_lattice_state.find(lattice_next_state);
						// We know this token exists in the output lattice (because it's in q_curr_frame_todo)
						KALDI_ASSERT(to_map_it != curr_f_raw_lattice_state.end());
						CostType token_extra_cost = to_map_it->second.token_extra_cost;
						StateId to_fst_lattice_state = to_map_it->second.fst_lattice_state;

						// We read the extra cost from lattice_next_state
						// We are now closing the state. If we write to it again, we will have to replay that frame
						// (so that the latest extra_cost value is read)
						to_map_it->second.is_state_closed = true;

						// We now need to consider all tokens related to that (iframe, fst_state)
						// with fst_state being the state this current token is linked to
						// There's two possibilies:
						// a) only one token is associated with that fst_state in that frame. The necessary information
						// is then stored directly in the token (arc_idx, prev_token)
						// b) multiple tokens are associated with that fst_state in that frame. The token that we have right now
						// only contains information on where to find the list of those tokens. It contains (offset, size)	
						// 
						// In any cases we consider the list of tokens to process as an array of InfoToken, which will 
						// be of size 1 in case a), of size > 1 in case b)
						InfoToken *tok_beg;
						float2 *arc_extra_cost_beg;
						int32 nprevs;
						if(token.IsUniqueTokenForStateAndFrame()) {
							tok_beg = &token;
							// if we've got only one, extra_cost == 0.0
							arc_extra_cost_beg = NULL; 
							nprevs = 1;
						} else {
							int32 offset, size;
							std::tie(offset,size) = token.GetNextStateTokensList();
							tok_beg = &h_all_tokens_extra_prev_tokens_[ichannel][offset];
							arc_extra_cost_beg = &h_all_tokens_extra_prev_tokens_extra_cost_[ichannel][offset];
							nprevs = size; 
						}
						
						// Used as a debugging tool. Each (iframe,fst_state) must have at least one token
						// with arc_extra_cost == 0.0f
						bool dbg_found_zero = false;
						for(int32 iprev=0; iprev<nprevs; ++iprev) {
							InfoToken list_token = tok_beg[iprev];
							int32 list_prev_token_idx = list_token.prev_token;
							int32 list_arc_idx = list_token.arc_idx;

							CostType arc_extra_cost;
							CostType acoustic_cost;
							if(arc_extra_cost_beg) {
								float2 both = arc_extra_cost_beg[iprev];
								arc_extra_cost = both.x;
								acoustic_cost = both.y;
							} else {
								// If we have only one token for that (iframe,fst_state),
								// Its arc has an extra_cost of zero (it's the only way to
								// get to that state, so it's the best)
								arc_extra_cost = 0.0f;
								acoustic_cost = h_all_tokens_acoustic_cost_[ichannel][token_idx];
							}
							// If we use that arc to go to prev_token, prev_token will have the 
							// following extra cost	
							CostType this_arc_prev_token_extra_cost = token_extra_cost + arc_extra_cost;
							// We need at least one arc_extra_cost of zero for each (iframe, fst_state)
							// The only use for that boolean is in a KALDI_ASSERT,
							// because if something went wrong in the kernels it's not likely that 
							// this property will be verified out of luck
							dbg_found_zero |= (arc_extra_cost == 0.0f);

							// Having the predecessor in the previous frame
							// <=> that token is associated to an emiting arc
							bool emitting = (list_prev_token_idx < curr_frame_offset);
							InfoToken list_prev_token = h_all_tokens_info_[ichannel][list_prev_token_idx];
							// Source of the arc currently considered
							StateId lattice_src_state = (list_prev_token_idx != 0) 
										? get_unique_id(list_prev_token_idx, list_prev_token)
										: fst_lattice_start; 
							//bool keep_arc = (arc_extra_cost == 0.0f); for one best
							// We only keep the arc if, when using that arc, we can end up
							// at the last frame with a cost not worse than (best+lattice_beam)
							// this_arc_prev_token_extra_cost contains the accumulated sums
							// of extra costs (through the cheapest possible way) to the last frame
							bool keep_arc = (this_arc_prev_token_extra_cost < lattice_beam_);

							if(keep_arc) {
								// We will now add this arc to the output lattice
								// We now the destination state of the arc (lattice_next_state)
								// has already been added to the output lattice 
								// (because its in q_curr_frame_todo)
								// We may need to add the source of that arc to the output fst
								StateId from_fst_lattice_state;
								if(list_prev_token_idx != 0) {
									// Selecting the right map
									// - emitting arc -> previous frame map
									// - non emitting arc -> same frame map
									auto *extra_cost_map = emitting ? &prev_f_raw_lattice_state : &curr_f_raw_lattice_state;
									decltype(extra_cost_map->end()) from_map_it;
									bool inserted;
									// Attempting to insert the state in the map
									std::tie(from_map_it, inserted) = extra_cost_map->insert({lattice_src_state, {FLT_MAX,-1,false}});
									// If it was inserted, its the first time we insert that key in the map
									// we need to put that state in the todo list to be considered next
									if(inserted) {
										auto *todo_list = emitting ? &q_prev_frame_todo : &q_curr_frame_todo;
										todo_list->push_back({list_prev_token_idx,list_prev_token});
										from_map_it->second.fst_lattice_state = fst_out->AddState();
									} 

									// Updating the source extra cost using that arc
									// for an arc a->b
									// extra_cost(a) = min(extra_cost(a),
									//		extra_cost(b) + arc_extra_cost(a->b))
									CostType prev_token_extra_cost = from_map_it->second.token_extra_cost;
									if(this_arc_prev_token_extra_cost < prev_token_extra_cost) {
										// We found a new min
										CostType diff = (prev_token_extra_cost - this_arc_prev_token_extra_cost);
										// If the change is large enough,
										// and if the state that we're writing to was already closed,
										// then we need to replay that frame.
										// if the source state is already closed it means we've
										// read its extra_cost value. Now we're writing again to it.
										// We have to do the first read again, to get the updated value
										// that's why we're replaying that frame 
										// (between frames everything is in topological order)
										if(diff >= min_delta && from_map_it->second.is_state_closed) {
											must_replay_frame = true;
										}
										prev_token_extra_cost = this_arc_prev_token_extra_cost;
									}

									// TODO put in above if
									from_map_it->second.token_extra_cost = prev_token_extra_cost;
									// Reading the StateId of the source state in the output lattice
									from_fst_lattice_state = from_map_it->second.fst_lattice_state;
								} else {
									from_fst_lattice_state = fst_lattice_start;
								}

								// Checking if it's the first time we insert an arc with that arc_idx, 
								// for that frame. 
								// If we're replaying that frame, we don't want duplicates
								bool is_this_arc_new = f_arc_idx_added.insert(list_arc_idx).second;
								if(is_this_arc_new) {
									// The following reads will most likely end up in cache misses
									// we could load everything sooner 
									LatticeArc arc(fst_.h_arc_id_ilabels_[list_arc_idx], 
											fst_.h_arc_olabels_[list_arc_idx],
											LatticeWeight(fst_.h_arc_weights_[list_arc_idx], acoustic_cost),
											to_fst_lattice_state);
									fst_out->AddArc(from_fst_lattice_state, arc);
								}
							}
 						}
						KALDI_ASSERT(dbg_found_zero);
					}

					if(must_replay_frame) {
						// The case described above 
						// (in the "if(this_arc_prev_token_extra_cost < prev_token_extra_cost)")
						// was triggered
						// We need to replay the frame. Because all states will be read again,
						// we can reopen them (and they will be closed again when being read from again)
						for(auto it = curr_f_raw_lattice_state.begin();
							 it != curr_f_raw_lattice_state.end();
							 ++it) {
							// Reopening state, we're going to replay the frame
							// The reads that closed the state in the past will be done again
							it->second.is_state_closed = false;
						}
					}
				} while(must_replay_frame);

				q_prev_frame_todo.swap(q_curr_frame_todo);
				q_prev_frame_todo.clear();
				prev_f_raw_lattice_state.swap(curr_f_raw_lattice_state);
				prev_f_raw_lattice_state.clear();
				f_arc_idx_added.clear();

				KALDI_ASSERT(q_prev_frame_todo.empty());
				if(iframe > 1)
					KALDI_ASSERT(!q_curr_frame_todo.empty());

			}

			nvtxRangePop();
		}	
		return true;
	}	

	void CudaDecoder::SetChannelsInKernelParams(const std::vector<ChannelId> &channels) {
		KALDI_ASSERT(channels.size() <= nchannels_);
		KALDI_ASSERT(channels.size() <= nlanes_);
		for(LaneId lane_id=0; lane_id<channels.size(); ++lane_id)
			h_kernel_params_->channel_to_compute[lane_id] = channels[lane_id];
		h_kernel_params_->nlanes_used = channels.size();
	}

	int32 CudaDecoder::NumFramesDecoded(ChannelId ichannel) const {
		KALDI_ASSERT(ichannel < nchannels_);
		return num_frames_decoded_[ichannel];	
	}
/*
	int32 CudaDecoder::NumFramesDecoded() const {
		return NumFramesDecoded(0);
	}
*/
} // end namespace kaldi.
