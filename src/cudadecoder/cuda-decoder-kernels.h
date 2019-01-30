#ifndef KALDI_DECODER_CUDA_DECODER_KERNELS_H_
#define KALDI_DECODER_CUDA_DECODER_KERNELS_H_

#define KALDI_CUDA_DECODER_HASHMAP_NO_KEY -1

#include "cuda-decoder.h"

namespace kaldi {
	__global__ void get_best_cost_kernel(DeviceParams cst_dev_params,KernelParams params, bool isfinal, CostType fst_zero);
	__global__ void finalize_process_non_emitting_kernel(DeviceParams cst_dev_params,KernelParams params);
	__global__ void exclusive_sum_batched_step3_kernel(DeviceParams cst_dev_params,KernelParams params);
	__global__ void exclusive_sum_batched_step2_kernel(DeviceParams cst_dev_params,KernelParams params);
	__global__ void save_channels_state_from_lanes_kernel(DeviceParams cst_dev_params,KernelParams params);
	__global__ void load_channels_state_in_lanes_kernel(DeviceParams cst_dev_params,KernelParams params);
	__global__ void init_decoding_on_device_kernel(DeviceParams cst_dev_params,KernelParams params);
	__global__ void initialize_initial_lane_kernel(DeviceParams cst_dev_params);
	template<bool IS_EMITTING>
		__global__ void expand_arcs_kernel(DeviceParams cst_dev_params,KernelParams params);
	template<bool IS_EMITTING>
		__global__ void post_expand_kernel(DeviceParams cst_dev_params,KernelParams params);
	__global__ void preprocess_in_place_kernel(DeviceParams cst_dev_params,KernelParams params);
	__global__ void preprocess_and_contract_kernel(DeviceParams cst_dev_params,KernelParams params);
	template<typename T>
	__global__ void concatenate_lanes_data(DeviceParams cst_dev_params, KernelParams params, LaneMatrixInterface<T> src, T *concat);

	struct __align__(16) HashmapValueT { 
		// Map key : fst state
		int key;
		// Number of tokens associated to that state
		int count;
		// minimum cost for that state + argmin
		int2 min_and_argmin_int_cost;
	};

	__global__ void init_hashmap_kernel(DeviceParams cst_dev_params, KernelParams params);
	__global__ void fill_best_int_cost_kernel(DeviceParams cst_dev_params,KernelParams params);
	__global__ void fill_extra_prev_tokens_list_kernel(DeviceParams cst_dev_params, KernelParams params);
	__global__ void clear_hashmap_kernel(DeviceParams cst_dev_params, KernelParams params);
	__global__ void compute_costs_histogram_kernel(DeviceParams cst_dev_params, KernelParams params, bool use_aux_q);
	__global__ void update_beam_using_histogram_kernel(DeviceParams cst_dev_params,KernelParams params, bool use_aux_q);

	__global__ void get_best_cost_kernel_step1(DeviceParams cst_dev_params,KernelParams params, bool use_final_probs, CostType fst_zero);
	__global__ void get_best_cost_kernel_step2(DeviceParams cst_dev_params,KernelParams params, bool use_final_probs, CostType fst_zero);

	template<typename T> 
		struct LaneMatrixInterface  {
			T *data_;	
			int32 ld_;	 // leading dimension - may use a log2 at some point
			__device__ __inline__ T *lane(const int32 ilane) {
				return &data_[ilane*ld_];
			}
		};

	template<typename T> 
		struct ChannelMatrixInterface {
			T *data_;	
			int32 ld_;	 // leading dimension
			__device__ __inline__ T *channel(const int32 ichannel) {
				return &data_[ichannel*ld_];
			}
		};

	struct DeviceParams {
		ChannelMatrixInterface<ChannelCounters> d_channels_counters; 
		LaneMatrixInterface<LaneCounters> d_lanes_counters; 

		ChannelMatrixInterface<int2> d_main_q_state_and_cost; 
		LaneMatrixInterface<CostType> d_main_q_acoustic_cost; 
		LaneMatrixInterface<InfoToken> d_main_q_info; 

		LaneMatrixInterface<int2> d_aux_q_state_and_cost; // TODO int_cost
		LaneMatrixInterface<CostType> d_aux_q_acoustic_cost; 
		LaneMatrixInterface<InfoToken> d_aux_q_info; 
		ChannelMatrixInterface<int32> d_main_q_arc_offsets; 
		LaneMatrixInterface<HashmapValueT> d_hashmap_values; 

		LaneMatrixInterface<int2> d_list_final_tokens_in_main_q; 

		LaneMatrixInterface<float2> d_main_q_extra_cost; 

		LaneMatrixInterface<int32> d_histograms; 

		ChannelMatrixInterface<int32> d_main_q_degrees_prefix_sum; 
		LaneMatrixInterface<int2> d_main_q_block_sums_prefix_sum; 
		LaneMatrixInterface<int32> d_main_q_representative_id; 
		LaneMatrixInterface<int32> d_main_q_extra_prev_tokens_prefix_sum;
		LaneMatrixInterface<int32> d_main_q_n_extra_prev_tokens_local_idx;
		LaneMatrixInterface<InfoToken> d_main_q_extra_prev_tokens;
		int32 max_nlanes;
		// TODO use the CudaFst struct
		int32 q_capacity;
		CostType *d_arc_weights;
		int32 *d_arc_nextstates;
		int32 *d_arc_pdf_ilabels;
		uint32 *d_arc_e_offsets;
		uint32 *d_arc_ne_offsets;
		CostType *d_fst_final_costs;
		int32 nstates;
		CostType default_beam;
		bool generate_lattices;
		CostType lattice_beam;
		int32 init_channel_id;
		StateId init_state; 
		CostType init_cost;
		int32 hashmap_capacity;
		int32 max_active;	
	};

	// TODO add STATIC_ASSERT for this struct size < 4KB
	struct KernelParams {
		// In AdvanceDecoding,
		// the lane lane_id will compute the channel
		// with channel_id = channel_to_compute[lane_id]
		ChannelId channel_to_compute[KALDI_CUDA_DECODER_MAX_N_LANES];
		int32 main_q_end_lane_offsets[KALDI_CUDA_DECODER_MAX_N_LANES];
		BaseFloat *loglikelihoods_ptrs[KALDI_CUDA_DECODER_MAX_N_LANES];
		int32 nlanes_used;
	};

	int32 floatToOrderedIntHost(float floatVal);
	float orderedIntToFloatHost(int32 intVal);

	typedef unsigned char BinId;	
} // namespace kaldi
#endif
