// decoder/cuda-decoder-kernels.cu
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

#include "cuda-decoder-kernels.h"
#include <cub/cub.cuh>

#define KALDI_CUDA_DECODER_DIV_ROUND_UP(a,b) ((a+b-1)/b)
namespace kaldi {
	// 1:1 Conversion float <---> sortable int
	// We convert floats to sortable ints in order
	// to use native atomics operation, which are 
	// way faster than looping over atomicCAS 
	__device__ __forceinline__ int32 binsearch_maxle(const int32 *vec, const int32 val, int32 low, int32 high) {
		while(true) {
			if(low == high)
				return low; //we know it exists
			if((low + 1) == high)
				return (vec[high] <= val) ? high : low;

			int32 mid = low + (high- low) / 2;

			if(vec[mid] > val)
				high = mid-1;
			else
				low = mid;
		}
	}

	__device__ int32 floatToOrderedInt(float floatVal) {
		int32 intVal = __float_as_int( floatVal );
		return (intVal >= 0 ) ? intVal : intVal ^ 0x7FFFFFFF;
	}


	__device__ float orderedIntToFloat(int32 intVal) {
		return __int_as_float( (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF );
	} 

	int32 floatToOrderedIntHost(float floatVal) {
		int32 intVal = reinterpret_cast<int&>( floatVal );
		return (intVal >= 0 ) ? intVal : intVal ^ 0x7FFFFFFF;
	}


	float orderedIntToFloatHost(int32 intVal) {
		intVal =  (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF;
		return reinterpret_cast<float&>(intVal);
	} 

	struct MinPlus {
		__device__ int2 operator()(const int2 &a, const int2 &b) const {
			int2 c;
			c.x = min(a.x, b.x);
			c.y = a.y + b.y;
			return c;
		}
	};

	struct PlusPlus {
		__device__ int2 operator()(const int2 &a, const int2 &b) const {
			int2 c;
			c.x = a.x + b.x;
			c.y = a.y + b.y;
			return c;
		}
	};

	union UInt64UnionInt2{
		int2 i2;
		unsigned long long int ull;
	};

	__device__ __inline__ int2 atomicAddI2(int2 *ptr, int2 val) {
		unsigned long long int *ptr64 = reinterpret_cast<unsigned long long int*>(ptr);
		UInt64UnionInt2 uval, uold;
		uval.i2 = val;
		uold.ull = atomicAdd(ptr64, uval.ull); 
		return uold.i2;
	} 	

	// TODO use native atomicMin64
	__device__ __inline__ void atomicMinI2(int2 *ptr, int2 val) {
		unsigned long long int *ptr64 = reinterpret_cast<unsigned long long int*>(ptr);
		UInt64UnionInt2 old, assumed, value;
		old.ull = *ptr64;
		value.i2 = val;
		if(old.i2.x <= val.x) return;
		do {
			assumed = old;
			old.ull = atomicCAS(ptr64, assumed.ull, value.ull);
		} while(old.ull!=assumed.ull && old.i2.x > value.i2.x);
	}

	// GetAdaptiveBeam is used by ExpandArc and FinalizeProcessNonemitting
	//
	// Given the fact that the token queues are too small to store 
	// all possible tokens in the worst case scenario (where we could generate "nstates" tokens),
	// we need to tighten the beam if we notice that we are at risk of overflowing either the aux_q
	// or the main_q

	__device__ void UpdateAdaptiveBeam(const DeviceParams &cst_dev_params, 
							const int aux_q_index_block_offset, 
							IntegerCostType min_int_cost,
							int2 *adaptive_int_beam_with_validity_index,
							LaneCounters *lane_counters) {
		int32 beam_valid_until_idx = adaptive_int_beam_with_validity_index->y;
		if(aux_q_index_block_offset < beam_valid_until_idx) 
			return; //nothing to do

		CostType beam = orderedIntToFloat(adaptive_int_beam_with_validity_index->x);
		while(aux_q_index_block_offset >= beam_valid_until_idx) {
			beam /= 2;
			beam_valid_until_idx += cst_dev_params.adaptive_beam_bin_width;
		}
		IntegerCostType new_int_cutoff = floatToOrderedInt(orderedIntToFloat(min_int_cost) + beam);
		IntegerCostType int_beam = floatToOrderedInt(beam);
		adaptive_int_beam_with_validity_index->x = int_beam;
		adaptive_int_beam_with_validity_index->y = beam_valid_until_idx; 
		// We can have races between the two atomics
		// However the worst than can happen is a CTA might delay updating the beam
		// This is not a critical bug. However, once we have a floatToOrderedInt
		// that is generating unsigned ints, we could merge the two atomics into a single atomic64
		atomicMin(&lane_counters->adaptive_int_beam_with_validity_index.x, int_beam);
		atomicMax(&lane_counters->adaptive_int_beam_with_validity_index.y, beam_valid_until_idx);
		atomicMin(&lane_counters->int_cutoff, new_int_cutoff);
	}


	//
	// HASHMAP
	//

	__device__ int hash_func(int key) {
		return key; // TODO
	}

	__global__ void init_hashmap_kernel(DeviceParams cst_dev_params, KernelParams params) {
		const int max_nlanes = cst_dev_params.max_nlanes;
		KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, max_nlanes) {
			const int capacity = cst_dev_params.hashmap_capacity;
			KALDI_CUDA_DECODER_1D_KERNEL_LOOP(idx, capacity) {
				cst_dev_params.d_hashmap_values.lane(ilane)[idx] = {KALDI_CUDA_DECODER_HASHMAP_NO_KEY, 0, {INT_MAX, -1}}; 
			}
		}
	}

	__device__ __forceinline__ HashmapValueT hashmap_find(HashmapValueT *d_map_values, int key, int capacity, int *hash_idx2val) {
		int hash_idx = hash_func(key)%capacity;
		int c = 0;	

		do {
			HashmapValueT val = d_map_values[hash_idx];
			if(val.key == key) {
				*hash_idx2val = hash_idx;
				return val;
			}
			if(val.key == KALDI_CUDA_DECODER_HASHMAP_NO_KEY)
				break;

			hash_idx = (hash_idx+1)%capacity;
			++c;
		} while(c < capacity);
		
		*hash_idx2val = NULL;
		return {KALDI_CUDA_DECODER_HASHMAP_NO_KEY, 0, {INT_MAX, -1}}; // no such key in map 
	}

	__device__ void hashmap_insert(HashmapValueT *d_map_values, int key, int int_cost, int arg, int capacity, int *local_idx) {
		int hash_idx = hash_func(key)%capacity;
		int c = 0;	
		HashmapValueT *d_val = NULL;
		do {
			d_val = &d_map_values[hash_idx];
			int old = atomicCAS(&d_val->key, KALDI_CUDA_DECODER_HASHMAP_NO_KEY, key);
			if(old == KALDI_CUDA_DECODER_HASHMAP_NO_KEY || old == key)
				break; // found a spot
			hash_idx = (hash_idx+1)%capacity;
			++c;
		} while(c < capacity);
		if(!d_val) 
			return;

		// Updating values
		*local_idx = atomicAdd(&d_val->count, 1); 
		atomicMinI2(&d_val->min_and_argmin_int_cost, {int_cost, arg});
	}


	/*
	   This kernel preprocess the necessary information for expand (scan of the outgoing degrees) 
	   and explicitly prune the tokens

	   The ExpandArc kernel writes the new raw token list in the aux_q. However, the cutoff 
	   was progressively lowered during the computation, and some tokens now have a cost > cutoff.
	   During the contract stage of this kernel, we remove such tokens. 
	   We also remove duplicates, i.e. tokens pointing to the same state, but with token.cost > best_cost_for_that_state

	   It contracts (by pruning) the queue list:
	   raw output in aux_q ----contract----> pruned output in main q

	   This kernel is responsible for :

	   1) Read a token from the aux queue (raw output from previous expand)

	   2) Compute the outgoing degree of that token.next_state. For that :
	   -> If that token is suboptimal (cutoff, best_cost), we prune it
	   -> Otherwise, we will move it to the main_q. We also read its arc degree in the FST graph 

	   3) We move the non-pruned tokens into the main q. After a local prefix sum,
	   we request a spot in the main_q for those tokens using the main_q_end_and_narcs counter. 
	   main_q_end_and_narcs.split.end contains the number of tokens in the main q until now
	   main_q_end_and_narcs.split.narcs contains the number of arcs in the main q until now

	   We also compute the degrees prefix sum in one pass using the main_q_end_and_narcs.split.narcs
	 */

	// Important : pass the struct PreprocessParams by copy - passing it using a ref will not work (CPU -> GPU)
	__global__ void preprocess_and_contract_kernel(DeviceParams cst_dev_params,KernelParams params) {
		typedef cub::BlockScan<int2, KALDI_CUDA_DECODER_1D_BLOCK> BlockScan;
		__shared__ typename BlockScan::TempStorage sh_temp_storage;
		// We need to move the survival tokens to the main_q
		// 
		// sh_main_q_global_block_offset has two purposes :
		// (1) to know where to store the survival tokens in the main_q
		// (2) to perform the prefix sum degrees of the survival degrees
		//
		// The reason why we store those two values together is because they are linked (see below)
		//
		// (1) We need a spot to store those tokens in the main_q 
		// We will ask the main_q counter where to store those tokens, the answer will be 
		// an offset of the main_q. We will store our tokens in positions :
		// d_main_q_state[sh_main_q_global_block_offset.ntokens], d_main_q_state[sh_main_q_global_block_offset.ntokens+1]...
		//
		// (2) sh_main_q_global_block_offset.narcs contains the number of arcs in the main_q up until index sh_main_q_global_block_offset.ntokens
		// ie the number of arcs going out of all states in d_main_q_state[0..sh_main_q_global_block_offset.ntokens]
		// it is used to compute the global prefix sum of degrees in one pass
		//
		__shared__ int2 sh_main_q_global_block_offset;
		const int nlanes = params.nlanes_used;
		KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
			// The condition of the for loop is the same for all threads in the CUDA block
			// we want to keep all threads alive at the same time for now
			// otherwise __syncthreads() would fail
			LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
			const int32 aux_q_end = lane_counters->post_expand_aux_q_end;
			const IntegerCostType int_cutoff = lane_counters->int_cutoff;
			KALDI_CUDA_DECODER_1D_BLOCK_OFFSET_KERNEL_LOOP(block_offset, thread_idx, aux_q_end) {
				const int32 aux_q_idx = block_offset + thread_idx;
				const int32 ichannel = params.channel_to_compute[ilane];
				int32 degree = 0;
				int32 arc_start = -1;
				StateId token_state;
				IntegerCostType token_int_cost;
				if(aux_q_idx < aux_q_end) {
					// Cost and state associated with the token
					const int2 both = cst_dev_params.d_aux_q_state_and_cost.lane(ilane)[aux_q_idx];
					token_state = both.x;
					token_int_cost = both.y;
					// Best cost for that token_state
					// We know we have a token associated with token_state in the queue with the cost state_best_cost
					// Final cutoff from last ExpandArc execution
					// Cutoff may have decreased since the creation of the token
					if(token_int_cost < int_cutoff) {
						// If generating lattices, keeping the token as long as its not too bad compared to best path 
						// to that state
						// If not generating lattices (one-best), keeping the token if its the best for that state
						// Contract is always called for non-emitting
						// using non-emitting offsets
						arc_start = cst_dev_params.d_arc_ne_offsets[token_state];
						const int32 arc_end = cst_dev_params.d_arc_ne_offsets[token_state+1];
						degree = arc_end - arc_start;
					}
				}

				int32 is_pruned = (arc_start == -1);
				// We now know which tokens will be moved to the main_q, the remaining will be pruned
				// we now compute a prefix sum inside the CUDA block to determine the local indexes of the survival tokens
				// the first survival token will have a index of 0, the second 1, ...
				// We also need to compute the prefix sum of the degrees
				// we start by doing a local prefix sum inside the CUDA block
				int2 block_prefix_sum_narcs_and_end = {degree, (is_pruned ? 0 : 1)};
				const int2 zero2 = {0,0};

				// Computing the prefix sum (exclusive)
				BlockScan(sh_temp_storage).ExclusiveScan(block_prefix_sum_narcs_and_end,
						block_prefix_sum_narcs_and_end, 
						zero2,
						PlusPlus());

				if(KALDI_CUDA_DECODER_IS_LAST_1D_THREAD()) {
					// This conditional branch is entered by the last thread
					// because it is the last, the prefix_sum of that thread contains the sum of all elts

					// We also add the value from this thread - the prefix sum is exclusive
					int2 block_sum = block_prefix_sum_narcs_and_end;
					block_sum.x += degree;
					block_sum.y += is_pruned ? 0 : 1;

					// Doing two things at the same time :
					// requesting a spot in the main_q to store the survival tokens from this CTA 
					// (we need space for token_and_arc_count_block_sum.split.ntokens tokens)
					// informing the main_q that our survival tokens contain token_arc_count_block_sum.split.narcs arcs
					//
					// We then store the return value, which is the global offset on where to store those tokens,
					// and the total number of arcs up until that global offset
					int2 block_offset = atomicAddI2(&lane_counters->main_q_narcs_and_end, block_sum);
					const int32 new_main_q_end = block_offset.y + block_sum.y;
					if(new_main_q_end >= cst_dev_params.q_capacity) {
						// We are overflowing the main_q
						// We first revert what this CTA has done, ie revert the previous atomicAdd
						// because all CTAs will revert, we know we will have a valid state after completion of this kernel
						//atomicSub(&lane_counters->main_q_end_and_narcs, block_sum); TODO
						lane_counters->q_overflow = 1; // for the host
						sh_main_q_global_block_offset.y = cst_dev_params.q_capacity; // used as flag to broadcast the information in the CTA 
						// We cannot jump to finalize_kernel now, we are in a divergent branch
					} else 
						sh_main_q_global_block_offset = block_offset;
				}

				// Syncing because : 
				// - Broadcasting sh_main_q_global_block_offset
				// - We may reuse sh_temp_storage (cf CUB doc)
				__syncthreads(); 

				// Checking if we are overflowing the main_q
				// All threads are executing the next line
				if(sh_main_q_global_block_offset.y == cst_dev_params.q_capacity) 
					return;	 //done

				// If we are executing the following lines it means that we are not overflowing the queue
				// We then continue what we were doing
				// Note : we could remove the branch divergence here 
				if(!is_pruned) {
					// This thread is in charge of a survival token
					// we will move it to the main_q, at index main_q_idx
					const int32 main_q_idx = sh_main_q_global_block_offset.y + block_prefix_sum_narcs_and_end.y;
					// Moving the token to the main q
					cst_dev_params.d_main_q_state_and_cost.channel(ichannel)[main_q_idx] = {token_state, token_int_cost};
					cst_dev_params.d_main_q_info.lane(ilane)[main_q_idx] = cst_dev_params.d_aux_q_info.lane(ilane)[aux_q_idx];
					cst_dev_params.d_main_q_acoustic_cost.lane(ilane)[main_q_idx] = cst_dev_params.d_aux_q_acoustic_cost.lane(ilane)[aux_q_idx];
					// Saving the global prefix sum
					// = (narcs until now in the main queue) + (narcs until this thread in the CTA)
					const int32 prefix_sum_narcs = sh_main_q_global_block_offset.x + block_prefix_sum_narcs_and_end.x;
					cst_dev_params.d_main_q_degrees_prefix_sum.channel(ichannel)[main_q_idx] = prefix_sum_narcs;
					// Saving the CSR arc offset for that token's state
					// it will be used by the expand kernel, and avoid doing a new random memory access in the expand kernel
					cst_dev_params.d_main_q_arc_offsets.channel(ichannel)[main_q_idx] = arc_start;
				}
			}

		}
	}

	/*
	   PreprocessInPlace
	   This kernel is also a preprocessing kernel, but this time does it in place
	   ie it will not move tokens from the aux_q to the main_q
	   It will do the preprocess operation directly on the main_q
	   The tokens are already in the main q (they were placed here by a previous "contract and preprocess").

	   We cannot prune non-optimal tokens, because the tokens are already in the main_q (we cannot prune 
	   the main_q - it would break the prev_token indexes). To avoid doing unnecessary computation 
	   in the expand kernel, we simulate the pruning by setting non-optimal token's degree to 0
	   We then rely on the 1 thread = 1 arc exact load balacing of expand to ignore that token

	   Please note that even if 0 threads will perform work on an ignored token in expand (degree = 0),
	   it is not exactly the same as pruning it : the main_q accesses will not be perfectly coalesced
	   in expand, because some "dead" tokens exist between living ones

	   For the preprocess stage we have to compute the prefix sum of the tokens arc degrees
	   Here we have to do the prefix sum in two passes : first local prefix sums inside CUDA block,
	   then in a second kernel (finalize_preprocess_in_place), we add the necessary block offsets to end up 
	   with the global prefix sum

	   This preprocess step is used in ProcessEmitting. Tokens were placed in main_q by
	   the ProcessNonEmitting of the previous frame. We cannot renumber them (it would break
	   the prev_token index). We preprocess in place, leaving things as they are in main_q

	 */

	__global__ void preprocess_in_place_kernel(DeviceParams cst_dev_params,KernelParams params) {
		// Operator for the prefix sum inside the CUDA block
		typedef cub::BlockScan<int2, KALDI_CUDA_DECODER_1D_BLOCK> BlockScan;
		__shared__ typename BlockScan::TempStorage sh_temp_storage;

		const int nlanes = params.nlanes_used;
		KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
			const LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
			const int32 main_q_end = lane_counters->main_q_narcs_and_end.y;
			// The condition of the for loop is the same for all threads in the CUDA block
			// we want to keep all threads alive at the same time for now
			// otherwise __syncthreads() would fail
			KALDI_CUDA_DECODER_1D_BLOCK_OFFSET_KERNEL_LOOP(block_offset, thread_idx, main_q_end) {
				// Position of considered token in the main_q
				const int32 main_q_idx = block_offset + thread_idx; 
				const int32 ichannel = params.channel_to_compute[ilane];

				// Total number of arcs from that token's state
				int32 degree = 0; 
				// If a state has more than one predecessor, we will
				// list them in a separate list. They are associated to the representative for that state (cf below)
				int32 n_extra_prev_token = 0;
				bool representing_state = false;
				if(main_q_idx < main_q_end) {
					int2 both = cst_dev_params.d_main_q_state_and_cost.channel(ichannel)[main_q_idx]; 
					StateId token_state = both.x;
					IntegerCostType token_int_cost = both.y; 

					// Final cutoff from last ExpandArc execution
					// The cutoff can have decreased since moving tokens to the main_q
					// min_cost cannot be lower than before (we only did non-emitting phases since then)
					// but the adaptive beam may have lowered the beam
					const IntegerCostType int_cutoff = lane_counters->int_cutoff;

					// Best cost for that token_state
					// We know we have a token associated with token_state in the queue with the cost state_best_cost
					int hash_idx;
					HashmapValueT h_val = hashmap_find(cst_dev_params.d_hashmap_values.lane(ilane), token_state, cst_dev_params.hashmap_capacity, &hash_idx);
					const IntegerCostType state_best_int_cost_argmin = h_val.min_and_argmin_int_cost.y; // TODO for nonoptimal save that in nextstate
					bool is_representative = (state_best_int_cost_argmin == main_q_idx);
					cst_dev_params.d_main_q_representative_id.lane(ilane)[main_q_idx] = is_representative 
													? (-hash_idx-1)  // -1 to force it < 0
													: hash_idx;

					// One of the best token for that state will represent that state in the next frame
					representing_state = (main_q_idx == state_best_int_cost_argmin);
					if(representing_state) {
						if(token_int_cost < int_cutoff) {
							// Next step is emitting (next frame), using emitting offsets
							const int32 start = cst_dev_params.d_arc_e_offsets[token_state]; 
							const int32 end = cst_dev_params.d_arc_e_offsets[token_state+1]; 
							degree  = end - start;
							// Saving the start offset for the expand kernel
							// avoid a new random memory access
							cst_dev_params.d_main_q_arc_offsets.channel(ichannel)[main_q_idx] = start; // TODO put offset in next_state
						}
						n_extra_prev_token = (h_val.count > 1) 
								? (h_val.count) // in that case we move the list of predecessors somewhere else 
								: 0; // if only one predecessor, let's store it directly in the .prev_token element
					}
				}

				// Computing a local prefix sum inside that CUDA block
				// A second kernel will take care of adding the necessary offset to those local prefix sums

				int2 zeroi2 = {0,0};
				int2 vali2 = {degree, n_extra_prev_token};
				int2 aggi2;
				BlockScan(sh_temp_storage).ExclusiveScan(vali2, aggi2, zeroi2, PlusPlus());
				int32 degree_local_prefix_sum = aggi2.x;
				int32 n_extra_prev_token_prefix_sum = aggi2.y;

				if(main_q_idx < main_q_end) {
					// This is not the final global prefix sum
					// A second kernel will add the necessary offset
					cst_dev_params.d_main_q_degrees_prefix_sum.channel(ichannel)[main_q_idx] = degree_local_prefix_sum; 
					cst_dev_params.d_main_q_extra_prev_tokens_prefix_sum.lane(ilane)[main_q_idx] = n_extra_prev_token_prefix_sum; 
				}

				if(KALDI_CUDA_DECODER_IS_LAST_1D_THREAD()) {
					// Saving the local sum of degrees of that CUDA block
					// That's necessary to compute the global offset of that CUDA block,
					// and that offset is what we need to transform the local prefix sum into a global prefix sum
					const int local_sum_index = block_offset/KALDI_CUDA_DECODER_1D_BLOCK;
					// the prefix sum was exclusive, adding missing value
					const int degree_inclusive_sum = degree_local_prefix_sum + degree; 
					const int n_extra_prev_tokens_inclusive_sum = n_extra_prev_token_prefix_sum + n_extra_prev_token;
					cst_dev_params.d_main_q_block_sums_prefix_sum.lane(ilane)[local_sum_index] = {degree_inclusive_sum, n_extra_prev_tokens_inclusive_sum}; 
				}

				// Synchronization because: 
				// - we may need to reuse sh_temp_storage if the for loop iterates (cf CUB's doc)
				__syncthreads(); 
			}
		}
	}

	//
	// ExpandArc kernel
	// This kernel does the actual work of traversing arcs 
	//
	// Pseudo code :
	// for all token tok in main_q[main_q_offset...end]:
	//      u = tok.next_state
	//      for all arc a(u->v) in the FST:
	//          v_cost = tok.cost + a.cost + accoustic_cost
	// 
	//          if v_cost < cutoff and v_cost < best_state_cost[v]
	//              generate token associated to v, add to aux_q
	//              update best_state_cost[v]
	//              if necessary update cutoff
	//
	// For more information please refer to http://kaldi-asr.org/doc/decoders.html
	//
	// ExpandArc rely on some preprocessed data to be able to function 
	// for instance, it needs the prefix sum of the arc degree of all token.state in the
	// main_q
	// We need to call a Preprocess kernel before ExpandArc
	//
	// ExpandArc is used for both emitting and nonemitting phases
	// Differences between emitting and nonemitting :
	//      1) params.d_q_arc_offset contains offsets to either emitting or nonemitting arcs. 
	//         It is transparent for this kernel. The differentiation was done in the Preprocess kernel,
	//         which is responsible for filling the params.d_q_arc_offset array
	//      2) Computation of the acoustic cost. If nonemitting, it is equal to 0. If emitting, we need
	//         to use values from the acoustic model (through the d_loglikelihoods array)
	//
	//
	//
	// Note : ExpandArc is not the only kernel able to traverse arcs. 
	// FinalizeProcessNonemitting contains a simplified version of expand for only one CUDA block
	template<bool IS_EMITTING>
		__global__ void expand_arcs_kernel(DeviceParams cst_dev_params, KernelParams params) {
			// BlockScan that we will use to compute token indexes in the output queue, 
			// and to find the min cost in the block
			typedef cub::BlockScan<int2, KALDI_CUDA_DECODER_1D_BLOCK> BlockScan;
			__shared__ typename BlockScan::TempStorage sh_temp_storage_scan;

			// This kernel writes the new token to the output queue aux_q
			// We will request a spot to store all the new tokens created by threads in this CUDA block
			// sh_aux_q_index_block_offset indicates where to store them in the aux_q
			// tokens created in this CUDA block will be store in :
			// aux_q[sh_aux_q_index_block_offset], aux_q[sh_aux_q_index_block_offset + 1], ...
			__shared__ int32 sh_aux_q_index_block_offset;
			const int nlanes = params.nlanes_used;
			KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
				LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
				const int32 main_q_offset = lane_counters->main_q_local_offset;
				const int32 main_q_end = lane_counters->main_q_narcs_and_end.y;
				const int32 total_narcs = lane_counters->main_q_narcs_and_end.x;
				KALDI_CUDA_DECODER_1D_BLOCK_OFFSET_KERNEL_LOOP(block_offset, thread_idx, total_narcs) {
					int2 adaptive_int_beam_with_validity_index = lane_counters->adaptive_int_beam_with_validity_index;
					// Position of considered token in the main_q
					const int32 ichannel = params.channel_to_compute[ilane];
					// Important : this thread is not responsible for a token in the input queue main_q
					// but for an arc, going out of a token in the main_q
					// The main_q contains in total total_narcs
					// and this thread will compute the main_q_arc_index-th arc of the main_q
					// For instance, first thread in the grid with threadIdx.x == 0 and blockIdx.x == 0 
					// will process the first arc of the token in main_q[main_q_offset + 0] 
					// (if that token has at least one arc)
					//
					// This insure a perfect one thread = one arc load balancing
					// but we have work to do to know exactly which arc is the main_q_arc_index-th arc
					// (what's its source ? its destination ? its arc_idx the FST CSR ?)
					int32 main_q_arc_index = block_offset + thread_idx;
					// We'll need those variables later in the kernel
					// we declare them outside of the "valid_input" scope
					// to be able to access them later
					int32 main_q_idx;
					int32 arc_idx;
					StateId arc_next_state;
					IntegerCostType int_total_cost = INT_MAX;
					CostType acoustic_cost = 0.0f;
					if(main_q_arc_index < total_narcs) {
						// Current thread must take care of main_q_arc_index-th arc
						// we need to now what's the source of that arc
						// ie which token.state in main_q does it start from ? 
						// We use a binary search in the prefix sum of the token's degree to get that information
						// 
						// Example : main_q contains 3 tokens
						// - First token is associated to a state which has 3 outgoing arc
						// - Second token is associated to a state which has 0 outgoing arc
						// - Third token is associated to a state which has 2 outgoing arc
						//
						// We store the degrees in an array :
						// [3, 0, 2]
						//
						// We then compute the exclusive prefix sum of that array :
						// [0, 3, 3, 5]
						//
						// In total, we have 5 arcs in the main_q. ExpandArc will use 5 threads.
						//
						// Let's say we are the fifth thread in ExpandArc. 
						// we have threadIdx.x == 4, and blockIdx.x == 0
						// it gives us main_q_arc_index == 4
						// From there we have no idea what we're supposed to do next, we need to have information about the
						// arc that we're supposed to traverse
						//
						// To do that, we look for the maximum index maxle_i in the prefix sum array such prefix_sum[i] <= 4
						//
						// [0, 3, 3, 5]
						//         /\
						//         here
						// maxle_i = 2
						// it means that our source token is at index 2 in the main_q
						// and we are computing the arc at index (main_q_arc_index - prefix_sum[maxle_i]) of that token 
						// ie the arc at index (4-3) = 1, the second arc of the second token in main_q

						// Searching for the source of the arc that we will process (main_q_arc_index)
						// we could preprocess the search in the preprocess kernels - for now this kernel is fast enough
						const int32 *degrees_prefix_sum = cst_dev_params.d_main_q_degrees_prefix_sum.channel(ichannel);
						main_q_idx = binsearch_maxle(degrees_prefix_sum, main_q_arc_index, main_q_offset, main_q_end-1); 

						// state_first_arc_idx_in_main_q
						// d_main_q_degrees_prefix_sum contains the prefix sum of the 
						// degrees of all tokens in the main_q
						// d_main_q_degrees_prefix_sum[main_q_idx] contains the number of arc
						// in the main_q until that token
						const int32 state_first_arc_idx_in_main_q = degrees_prefix_sum[main_q_idx];

						// arc_offset_start is the offset in the CSR, to find the arcs 
						// related to the state main_q_state_[main_q_idx]
						// it was set by the preprocess kernel
						const int32 arc_offset_start = cst_dev_params.d_main_q_arc_offsets.channel(ichannel)[main_q_idx];

						// local_arc_index is the arc index for that state
						// if local_arc_index == 2, we will process the second arc
						// of state main_q_state_[main_q_idx]
						const int32 local_arc_index = main_q_arc_index - state_first_arc_idx_in_main_q;

						// corresponding arc_idx in the FST
						arc_idx = arc_offset_start + local_arc_index; 

						// Destination of that arc
						arc_next_state = cst_dev_params.d_arc_nextstates[arc_idx];

						// Building the total cost incrementally 
						// we'll add the acoustic cost and the old token's cost
						const CostType arc_fixed_cost = cst_dev_params.d_arc_weights[arc_idx];
						const CostType prev_token_cost  = orderedIntToFloat(cst_dev_params.d_main_q_state_and_cost.channel(ichannel)[main_q_idx].y);
						CostType total_cost = prev_token_cost + arc_fixed_cost;
						const int32 prev_state = cst_dev_params.d_main_q_state_and_cost.channel(ichannel)[main_q_idx].x;
						if(IS_EMITTING) {
							const int32 arc_ilabel = cst_dev_params.d_arc_pdf_ilabels[arc_idx];
							acoustic_cost = -params.loglikelihoods_ptrs[ilane][arc_ilabel]; 
							total_cost += acoustic_cost;
						}
						int_total_cost = floatToOrderedInt(total_cost);

						// If the total_cost is too large compared to our cutoff (beam search)
						// then let's drop it
						const IntegerCostType int_cutoff = lane_counters->int_cutoff;
						if(int_total_cost >= int_cutoff)
							int_total_cost = INT_MAX;
					}

					//
					// If int_total_cost < INT_MAX, it means that : 
					// - this thread had a valid input (main_q_arc_index < total_narcs)
					// - the total_cost of the generated token is < cutoff
					// - the generated token is the best candidate for that next_state
					// We will then add that new token in the output queue, aux_q
					// We need to know where to put that token in the aux_q
					// we'll first compute its index inside the CUDA block
					// the first valid output token in the CUDA block will have index 0, 
					// the second index 1... We compute that using a prefix sum
					//
					// We also need to find the overall min cost in the CUDA block
					// a prefix sum is a scan operation, and a min a reduce operation
					// we can perform a reduce operation using a scan (using the last value)
					// we compute the prefix sum and the min in one scan, using the data 
					// struct CostTypeAndInt
					//
					const int32 has_successor = (int_total_cost < INT_MAX) ? 1 : 0; 

					int2 int_cost_and_index = {int_total_cost, has_successor};
					BlockScan(sh_temp_storage_scan).InclusiveScan(int_cost_and_index, int_cost_and_index, MinPlus());
					if(KALDI_CUDA_DECODER_IS_LAST_1D_THREAD()) {
						// We are in a divergent branch
						// This is the last thread. The last value of the inclusive scan is the total
						const int32 total_successors_in_block = int_cost_and_index.y;
						// Requesting a spot of size total_successors_in_block in the aux_q
						const int aux_q_index_block_offset = atomicAdd(&lane_counters->aux_q_end, total_successors_in_block);
						// We can find a lower global_min_cost only in the emitting stage
						if(IS_EMITTING) {
							IntegerCostType global_min_int_cost = lane_counters->min_int_cost;
							IntegerCostType local_min_int_cost = int_cost_and_index.x;
							// if we found a lower min_cost, update the global value
							if(local_min_int_cost < global_min_int_cost) {
								global_min_int_cost = local_min_int_cost;
								atomicMin(&lane_counters->min_int_cost, global_min_int_cost);
								CostType beam = orderedIntToFloat(adaptive_int_beam_with_validity_index.x);
								IntegerCostType new_int_cutoff = floatToOrderedInt(orderedIntToFloat(local_min_int_cost) + beam);
								atomicMin(&lane_counters->int_cutoff, new_int_cutoff);
							}
							int32 beam_valid_until_idx = adaptive_int_beam_with_validity_index.y;
							if(aux_q_index_block_offset >= beam_valid_until_idx) {
								// This beam is no longer valid. Updating it
								// TODO move out of emitting 
								UpdateAdaptiveBeam(cst_dev_params, 
										aux_q_index_block_offset, 
										global_min_int_cost, 
										&adaptive_int_beam_with_validity_index,
										lane_counters);
							}
						}

						// All threads will need this value
						// Saving in shared memory
						sh_aux_q_index_block_offset = aux_q_index_block_offset;
						//
						// Here we detect an overflow of the aux_q
						// we detect it before actually using the aux_q
						// We try to prevent an overflow from happening using an adaptive beam (cf GetAdaptiveBeam)
						//
						if((sh_aux_q_index_block_offset + total_successors_in_block) >= cst_dev_params.q_capacity) {
							// sh_aux_q_index_block_offset is in shared memory
							// its value is currently invalid (overflow)
							// we set it to a special value and use it as a flag to broadcast
							// the fact that we have an overflow and that all threads should exit
							sh_aux_q_index_block_offset = cst_dev_params.q_capacity;
							// We revert the last operation. All threads that detected the overflow 
							// will revert what they've done. It means that at the end of the kernel,
							// we'll be back to the last valid state 
							// We'll be able to continue computation, but quality of the output
							// may be lower (we weren't able to save all tokens)
							atomicAdd(&lane_counters->aux_q_end, -total_successors_in_block); 
							// Setting the flag for the host. It will be used to print a warning to stderr
							lane_counters->q_overflow = 1; 
							// We do not jump to finalize_kernel now, because only threadIdx.x == 0 
							// is executing this
							// We wait until the end of the divergent branch
						}
					}

					// Sync'ing for two reasons :
					// - Broadcasting sh_aux_q_index_block_offset
					// - reusing sh_temp_storage (cf CUB's doc)
					__syncthreads(); 
					// The only case where we can have that condition met,
					// is if we detected an overflow if the previous lines
					// we need to finalize our work and quit 
					// Now all threads are executing this code. We can jump
					// to finalize_kernel
					if(sh_aux_q_index_block_offset == cst_dev_params.q_capacity) 
						return;
					//
					// If we're executing the following lines it means everything
					// is valid and we are not overflowing the aux_q
					//
					int_cost_and_index.y -= has_successor; // we want the exclusive sum now
					const int32 aux_q_block_index = int_cost_and_index.y;
					const int32 aux_q_index = sh_aux_q_index_block_offset + aux_q_block_index;
					if(has_successor) {
						// We save the new token to the aux_q
						cst_dev_params.d_aux_q_state_and_cost.lane(ilane)[aux_q_index] = {arc_next_state, int_total_cost};
						// Index of the parent token
						// the parent is the token used as input 
						// that parent is at index main_q_idx in the GPU memory
						// However, the main_q is emptied before processing a new frame
						// we need to add the offset related to the previous frames index
						// we add cst_dev_params.main_q_global_offset
						//if(IS_EMITTING) 
							cst_dev_params.d_aux_q_acoustic_cost.lane(ilane)[aux_q_index] = acoustic_cost;
						
						const int32 prev_token = lane_counters->main_q_global_offset + main_q_idx;
						cst_dev_params.d_aux_q_info.lane(ilane)[aux_q_index] = {prev_token, arc_idx};

					}
				}
			}
		}

	// Initialize initial channel
	// The initial channel is the state of a channel when 
	// it will start decoding a new utterance
	__global__ void initialize_initial_lane_kernel(DeviceParams cst_dev_params) {
		const int init_ichannel = cst_dev_params.init_channel_id;
		const int init_ilane = 0;
		ChannelCounters *init_channel_counters = cst_dev_params.d_channels_counters.channel(init_ichannel);
		LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(init_ilane);

		lane_counters->aux_q_end = 0;
		lane_counters->post_expand_aux_q_end = 1;
		lane_counters->main_q_global_offset = 0; 
		lane_counters->main_q_local_offset = 0;
		lane_counters->main_q_n_extra_prev_tokens = 0;
		lane_counters->int_cutoff = INT_MAX;
		lane_counters->min_int_cost = INT_MAX;
		lane_counters->int_beam = floatToOrderedInt(cst_dev_params.default_beam);
    lane_counters->main_q_narcs_and_end = {0,0};

		// Simulate a previously generated aux_q containing init state
		const StateId init_state = cst_dev_params.init_state;
		const CostType init_cost = cst_dev_params.init_cost;
		IntegerCostType int_init_cost = floatToOrderedInt(init_cost);
		cst_dev_params.d_aux_q_state_and_cost.lane(init_ilane)[0] = {init_state, int_init_cost};
		cst_dev_params.d_aux_q_info.lane(init_ilane)[0] = {INT_MIN, -1};
	}


	// Called when channels will start decoding a new utterance
	// do everything that's needed to do on the device to start decoding a new utterance with those channels
	__global__ void init_decoding_on_device_kernel(DeviceParams cst_dev_params,KernelParams params) {
		const int init_ichannel = cst_dev_params.init_channel_id;
		const ChannelCounters *init_channel_counters = cst_dev_params.d_channels_counters.channel(init_ichannel);
		const int32 init_main_q_end = init_channel_counters->prev_main_q_narcs_and_end.y;
		const int32 nlanes = params.nlanes_used;
		KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
			KALDI_CUDA_DECODER_1D_KERNEL_LOOP(idx, init_main_q_end) { 
				const int32 ichannel = params.channel_to_compute[ilane];
				cst_dev_params.d_main_q_state_and_cost.channel(ichannel)[idx] = cst_dev_params.d_main_q_state_and_cost.channel(init_ichannel)[idx];
				cst_dev_params.d_main_q_degrees_prefix_sum.channel(ichannel)[idx] = cst_dev_params.d_main_q_degrees_prefix_sum.channel(init_ichannel)[idx];
				cst_dev_params.d_main_q_arc_offsets.channel(ichannel)[idx] = cst_dev_params.d_main_q_arc_offsets.channel(init_ichannel)[idx];
				if(idx == 0) {
					ChannelCounters *channel_counters = cst_dev_params.d_channels_counters.channel(ichannel);
					channel_counters->prev_main_q_narcs_and_end  = init_channel_counters->prev_main_q_narcs_and_end;
					channel_counters->prev_main_q_n_extra_prev_tokens = init_channel_counters->prev_main_q_n_extra_prev_tokens;
					channel_counters->prev_main_q_global_offset  = 0;
					channel_counters->prev_main_q_extra_prev_tokens_global_offset = 0;
					channel_counters->prev_beam  = cst_dev_params.default_beam;
				}
			}
		}
	}

	// Context switch : load
	// THREADS : (WARP_SIZE, 1, 1)
	// BLOCKS : (1, nchannel_to_compute, 1)
	__global__ void load_channels_state_in_lanes_kernel(DeviceParams cst_dev_params,KernelParams params) {
		const int nlanes = params.nlanes_used;
		KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
			const int32 ichannel = params.channel_to_compute[ilane];
			// Getting the lane ready for that channel
			LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
			const ChannelCounters *channel_counters = cst_dev_params.d_channels_counters.channel(ichannel);
			lane_counters->main_q_narcs_and_end = channel_counters->prev_main_q_narcs_and_end;
			lane_counters->main_q_n_extra_prev_tokens = channel_counters->prev_main_q_n_extra_prev_tokens; 
			CostType beam = channel_counters->prev_beam; // TODO rename prev_beam is actually the new frame beam
			IntegerCostType int_beam = floatToOrderedInt(beam);
			lane_counters->int_beam = int_beam; 
			lane_counters->adaptive_int_beam_with_validity_index.x = int_beam;
			lane_counters->adaptive_int_beam_with_validity_index.y = cst_dev_params.adaptive_beam_static_segment;
			lane_counters->main_q_global_offset = channel_counters->prev_main_q_global_offset; // we'll update it after emitting
 			lane_counters->main_q_extra_prev_tokens_global_offset = channel_counters->prev_main_q_extra_prev_tokens_global_offset;
			lane_counters->min_int_cost_and_arg_with_final.x = INT_MAX; // used by GetBestCost
			lane_counters->int_cutoff = INT_MAX;
			lane_counters->min_int_cost = INT_MAX;
			lane_counters->q_overflow = 0;	
		}
	}

	// Context switch : store
	__global__ void save_channels_state_from_lanes_kernel(DeviceParams cst_dev_params,KernelParams params) {
		const int nlanes = params.nlanes_used;
		KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
			const LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
			const int32 ichannel = params.channel_to_compute[ilane];
			ChannelCounters *channel_counters = cst_dev_params.d_channels_counters.channel(ichannel);
			channel_counters->prev_main_q_global_offset = lane_counters->main_q_global_offset;
 			channel_counters->prev_main_q_extra_prev_tokens_global_offset = lane_counters->main_q_extra_prev_tokens_global_offset;
			channel_counters->prev_main_q_narcs_and_end = lane_counters->main_q_narcs_and_end;
			channel_counters->prev_main_q_n_extra_prev_tokens = lane_counters->main_q_n_extra_prev_tokens; 
			channel_counters->prev_beam = orderedIntToFloat(lane_counters->int_beam);
		}
	}

	template<bool IS_EMITTING>
		__global__ void post_expand_kernel(DeviceParams cst_dev_params,KernelParams params) {
			const int nlanes = params.nlanes_used;
			KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
				LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
				const int prev_main_q_end = lane_counters->main_q_narcs_and_end.y;
				const int prev_n_extra_prev_tokens = lane_counters->main_q_n_extra_prev_tokens;
				const int aux_q_end = lane_counters->aux_q_end;
				// The next step is the contracting step from aux_q to main_q
				// It will need the aux_q_end value. But it will also empty the aux_q
				// We're resetting aux_q_end to 0 now, but we're saving its old value 
				// in another place
				lane_counters->post_expand_aux_q_end = aux_q_end;
				lane_counters->aux_q_end = 0;	
				// We are done processing those arcs
				lane_counters->main_q_narcs_and_end.x = 0;
 				// Resetting the adaptive beam
				lane_counters->adaptive_int_beam_with_validity_index.x = lane_counters->int_beam;
				lane_counters->adaptive_int_beam_with_validity_index.y = cst_dev_params.adaptive_beam_static_segment;
				if(IS_EMITTING) {
					// the main_q contains the tokens from the previous frame
					// after emitting, we won't use them anymore to create new tokens
					// we reset the main_q
					lane_counters->main_q_narcs_and_end = {0,0};
					// The main_q was flushed - we need to update the global_offset
					lane_counters->main_q_global_offset += prev_main_q_end;
					if(threadIdx.x == 0 && blockIdx.x == 0)
						lane_counters->main_q_extra_prev_tokens_global_offset += prev_n_extra_prev_tokens;
					// Moving local offset. Tokens created by last expand
					// will be pruned, and survivals will be moved at the end
					// of the main q. Those tokens will be placed after local_offset 
				} else {
					lane_counters->main_q_local_offset = prev_main_q_end;
				}
			}
		}

	// Batched scan is not available in CUB
	__global__ void exclusive_sum_batched_step2_kernel(DeviceParams cst_dev_params,KernelParams params) {
		typedef cub::BlockScan<int2, KALDI_CUDA_DECODER_1D_BLOCK> BlockScan;
		__shared__ typename BlockScan::TempStorage sh_temp_storage;
		const int nlanes = params.nlanes_used;
		KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
			LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
			const int main_q_end = lane_counters->main_q_narcs_and_end.y;
			const int ntiles = KALDI_CUDA_DECODER_DIV_ROUND_UP(main_q_end, KALDI_CUDA_DECODER_1D_BLOCK);
			// Using block_offset loop to keep entire CTA alive (we're going to use __syncthreads in CUB)
			int2 sum_so_far = {0,0};
			KALDI_CUDA_DECODER_1D_BLOCK_OFFSET_KERNEL_LOOP(offset, thread_idx, ntiles) {
				const int32 itile = offset + thread_idx;
				const int2 zeroi2 = {0,0};
				const int2 val = (itile < ntiles) 
						? cst_dev_params.d_main_q_block_sums_prefix_sum.lane(ilane)[itile] 
						: zeroi2;


				int2 prefix_sum, sum;
				BlockScan(sh_temp_storage).ExclusiveScan(val, prefix_sum, zeroi2, PlusPlus(), sum);
				PlusPlus pp;
				prefix_sum = pp(prefix_sum, sum_so_far);
				sum_so_far = pp(sum_so_far, sum);
				if(itile < ntiles) {
					cst_dev_params.d_main_q_block_sums_prefix_sum.lane(ilane)[itile] = prefix_sum;
				}
				if(itile == (ntiles-1)) {
					const int32 total_narcs = prefix_sum.x+val.x; 
					const int32 total_n_extra_prev_tokens = prefix_sum.y+val.y; 
					lane_counters->main_q_narcs_and_end.x = total_narcs;
					lane_counters->main_q_n_extra_prev_tokens = total_n_extra_prev_tokens;
				}

				if(itile == 0) {
					// Last time those were used was in previous kernel
					lane_counters->min_int_cost = INT_MAX;
					lane_counters->int_cutoff = INT_MAX;
					lane_counters->min_int_cost_and_arg_with_final.x = INT_MAX; // used by GetBestCost
					const CostType current_beam = orderedIntToFloat(lane_counters->int_beam);
					const CostType new_beam = fmin(cst_dev_params.default_beam, 
							current_beam*KALDI_CUDA_DECODER_ADAPTIVE_BEAM_RECOVER_RATE);
					lane_counters->int_beam = floatToOrderedInt(new_beam);
				}
			}
		}
	}

	// Batched scan is not available in CUB
	__global__ void exclusive_sum_batched_step3_kernel(DeviceParams cst_dev_params,KernelParams params) {
		const int nlanes = params.nlanes_used;
		KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
			const LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
			const int32 ichannel = params.channel_to_compute[ilane];
			const int main_q_end = lane_counters->main_q_narcs_and_end.y;
			KALDI_CUDA_DECODER_1D_KERNEL_LOOP(main_q_idx, main_q_end) {
				const int32 local_sum_idx = main_q_idx / KALDI_CUDA_DECODER_1D_BLOCK;
				const int2 local_sum_offset = cst_dev_params.d_main_q_block_sums_prefix_sum.lane(ilane)[local_sum_idx];
				int32 hash_idx = cst_dev_params.d_main_q_representative_id.lane(ilane)[main_q_idx]; // TODO rename

				cst_dev_params.d_main_q_degrees_prefix_sum.channel(ichannel)[main_q_idx] += local_sum_offset.x;
				int extra_prev_tokens_offset = cst_dev_params.d_main_q_extra_prev_tokens_prefix_sum.lane(ilane)[main_q_idx] + local_sum_offset.y; 
				bool is_representative = (hash_idx < 0);
				if(is_representative) {
					hash_idx = -hash_idx -1;
					HashmapValueT &val = cst_dev_params.d_hashmap_values.lane(ilane)[hash_idx];
					//printf("arg=%i idx=%i off=%i \n", val.min_and_argmin_int_cost.y, main_q_idx, extra_prev_tokens_offset);
					val.min_and_argmin_int_cost.y = extra_prev_tokens_offset;
				}
			}
		}
	}


	__global__ void fill_extra_prev_tokens_list_kernel(DeviceParams cst_dev_params, KernelParams params) {
		const int nlanes = params.nlanes_used;
		KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
			const LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
			const int32 ichannel = params.channel_to_compute[ilane];
			const int main_q_end = lane_counters->main_q_narcs_and_end.y;
			const int prev_global_idx = lane_counters->main_q_extra_prev_tokens_global_offset;
			KALDI_CUDA_DECODER_1D_KERNEL_LOOP(main_q_idx, main_q_end) {
				int32 hash_idx = cst_dev_params.d_main_q_representative_id.lane(ilane)[main_q_idx]; // TODO rename
				bool is_representative = (hash_idx < 0);
				if(is_representative) {
					hash_idx = -hash_idx -1; // reverting is_representative flag
				}
				HashmapValueT val = cst_dev_params.d_hashmap_values.lane(ilane)[hash_idx];
				int prev_counts = val.count;
				if(prev_counts > 1) { // in that case we must the info_token somewhere else
					CostType token_cost = orderedIntToFloat(cst_dev_params.d_main_q_state_and_cost.channel(ichannel)[main_q_idx].y); 
					CostType acoustic_cost = cst_dev_params.d_main_q_acoustic_cost.lane(ilane)[main_q_idx];
					InfoToken inf_tok = cst_dev_params.d_main_q_info.lane(ilane)[main_q_idx];
					CostType best_cost = orderedIntToFloat(val.min_and_argmin_int_cost.x);
					CostType extra_cost = token_cost - best_cost;
					int extra_prev_tokens_offset = val.min_and_argmin_int_cost.y;
					int local_idx = cst_dev_params.d_main_q_n_extra_prev_tokens_local_idx.lane(ilane)[main_q_idx]; 
					//printf("info[%i].prev=%i (local_idx=%i, go=%i, sum=%i) \n", main_q_idx, inf_tok.prev_token, local_idx, prev_global_idx, extra_prev_tokens_offset);
					cst_dev_params.d_main_q_info.lane(ilane)[main_q_idx] = {prev_global_idx+extra_prev_tokens_offset, -prev_counts}; // negative counts to signal that's a (offset,count) pair
					int list_idx = extra_prev_tokens_offset + local_idx;
					cst_dev_params.d_main_q_extra_prev_tokens.lane(ilane)[list_idx] = inf_tok; // moving the prev_tokens info in another list
					cst_dev_params.d_main_q_extra_cost.lane(ilane)[list_idx] = {extra_cost,acoustic_cost};
					//printf("extra_cost[%i] = %f \n", list_idx, extra_cost);
					//InfoToken inf2_tok = cst_dev_params.d_main_q_extra_prev_tokens.lane(ilane)[list_idx] ;
					//printf("info2[%i].prev=%i \n", list_idx, inf2_tok.prev_token);
				}
			}
		}
	}

	__global__ void clear_hashmap_kernel(DeviceParams cst_dev_params, KernelParams params) {
		const int nlanes = params.nlanes_used;
		KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
			const LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
			const int main_q_end = lane_counters->main_q_narcs_and_end.y;
			KALDI_CUDA_DECODER_1D_KERNEL_LOOP(main_q_idx, main_q_end) {
				int32 hash_idx = cst_dev_params.d_main_q_representative_id.lane(ilane)[main_q_idx];
				bool is_representative = (hash_idx < 0);
				if(is_representative) {
					hash_idx = -hash_idx -1; // reverting is_representative flag
					cst_dev_params.d_hashmap_values.lane(ilane)[hash_idx] =  {KALDI_CUDA_DECODER_HASHMAP_NO_KEY, 0, {INT_MAX, -1}}; // clear
				}
			}
		}
	}

	/*

	   FinalizeProcessNonemitting
	   Meta-kernel (merging preprocess and expand) but only works with 1 CUDA block

	   Used to avoid calling multiple "heavy lifting" kernels for the tail of non emitting
	   (lots of iterations with small number of arcs)

	   Code is greatly simplified because we can have only one CTA alive

	   Repeat until new queue empty:
	   1) Preprocess 
	   2) Expand arcs

	   The preprocess stage is not done on the first iteration, because it was
	   already done by the ProcessAndContract kernel. We always call ProcessAndContract
	   before calling FinalizeProcessNonemitting 

	   At the end, this kernel finalize the computation for current frame,
	   so that it's ready for next ProcessEmitting

Note : For a detailed description on how the Preprocess and Expand operation work,
please refer to the PreprocessInPlace and ExpandArc kernel implemention. The algorithm are 
described there. In this kernel, we compute simplified version of preprocess and expand, because
we do not need inter-block communication (we launch only one CUDA block)

	 */


	__launch_bounds__(KALDI_CUDA_DECODER_LARGEST_1D_BLOCK, 1)
		__global__ void finalize_process_non_emitting_kernel(DeviceParams cst_dev_params,KernelParams params) {
			typedef cub::BlockScan<int2, KALDI_CUDA_DECODER_LARGEST_1D_BLOCK> Int2BlockScan;
			typedef cub::BlockScan<int, KALDI_CUDA_DECODER_LARGEST_1D_BLOCK> IntBlockScan;
			__shared__ typename IntBlockScan::TempStorage sh_temp_storage_int_scan;
			__shared__ typename Int2BlockScan::TempStorage sh_temp_storage_int2_scan;

			const int nlanes = params.nlanes_used;
			KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
				LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
				const int32 ichannel = params.channel_to_compute[ilane];

				int2 both = lane_counters->main_q_narcs_and_end;
				int32 main_q_narcs = both.x;
				int32 main_q_end = both.y; 
				int32 main_q_local_offset = lane_counters->main_q_local_offset;
				const int32 main_q_global_offset = lane_counters->main_q_global_offset;
				// aux_q is empty when this kernel is called
				int32 aux_q_end = 0;
				IntegerCostType int_cutoff = lane_counters->int_cutoff;
				while(main_q_narcs > 0) {
					// Step 1 : ExpandArcs
					KALDI_CUDA_DECODER_1D_BLOCK_OFFSET_KERNEL_LOOP(offset, thread_idx, main_q_narcs)  {
						const int32 main_q_arc_idx = offset + thread_idx;
						// For details on how this code works, please refer to comments in expand_arcs
						IntegerCostType total_int_cost = INT_MAX;
						int32 arc_idx;
						StateId arc_next_state;
						int32 main_q_idx;
						if(main_q_arc_idx < main_q_narcs) {
							main_q_idx = binsearch_maxle(cst_dev_params.d_main_q_degrees_prefix_sum.channel(ichannel), 
									main_q_arc_idx, 
									main_q_local_offset,
									main_q_end-1); 

							const int32 state_first_arc_idx_in_main_q = cst_dev_params.d_main_q_degrees_prefix_sum.channel(ichannel)[main_q_idx];
							const int32 arc_offset_start = cst_dev_params.d_main_q_arc_offsets.channel(ichannel)[main_q_idx];
							arc_idx = arc_offset_start + (main_q_arc_idx - state_first_arc_idx_in_main_q);

							arc_next_state = cst_dev_params.d_arc_nextstates[arc_idx];
							CostType arc_weight = cst_dev_params.d_arc_weights[arc_idx];
							CostType prev_token_cost = orderedIntToFloat(cst_dev_params.d_main_q_state_and_cost.channel(ichannel)[main_q_idx].y); 
							total_int_cost = floatToOrderedInt(arc_weight + prev_token_cost);
							//const int32 prev_state = cst_dev_params.d_main_q_state_and_cost.channel(ichannel)[main_q_idx].x;

							if(total_int_cost >= int_cutoff) {
								total_int_cost = INT_MAX; // above cutoff 
							}

						}
						const int32 has_successor = (total_int_cost < INT_MAX) ? 1 : 0;

						int32 local_aux_q_idx;
						int32 nsuccessors;
						IntBlockScan(sh_temp_storage_int_scan).ExclusiveSum(has_successor, 
								local_aux_q_idx,
								nsuccessors); // aggregate

						// Checking if we are overflowing the aux_q
						if((aux_q_end + nsuccessors) >= cst_dev_params.q_capacity) {
							lane_counters->q_overflow = 1;
							// nothing to revert in global memory
							goto finalize_kernel;
						}

						if(has_successor) {
							const int32 aux_q_idx = aux_q_end + local_aux_q_idx;
							const int32 prev_token_idx = main_q_global_offset + main_q_idx;
							cst_dev_params.d_aux_q_state_and_cost.lane(ilane)[aux_q_idx] = {arc_next_state,total_int_cost};
							cst_dev_params.d_aux_q_info.lane(ilane)[aux_q_idx] = {prev_token_idx,arc_idx};
							cst_dev_params.d_aux_q_acoustic_cost.lane(ilane)[aux_q_idx] = 0.0f; // TODO remove
							//const int32 prev_state = cst_dev_params.d_main_q_state_and_cost.channel(ichannel)[main_q_idx].x;
							//const CostType prev_cost  = orderedIntToFloat(cst_dev_params.d_main_q_state_and_cost.channel(ichannel)[main_q_idx].y);
						}
						aux_q_end += nsuccessors;
						// reusing sh_temp_storage_scan_int TODO double buffering
						__syncthreads();
					}

					// Step 2 : PreprocessAndContract
					// Reset for new iteration
					main_q_narcs = 0;
					main_q_local_offset = main_q_end;
					KALDI_CUDA_DECODER_1D_BLOCK_OFFSET_KERNEL_LOOP(offset, thread_idx, aux_q_end) {
						const int32 aux_q_idx = offset + thread_idx;
						int32 degree = 0;
						int32 start = -1;
						StateId token_state;
						IntegerCostType token_int_cost;
						if(aux_q_idx < aux_q_end) {
							int2 both = cst_dev_params.d_aux_q_state_and_cost.lane(ilane)[aux_q_idx];
							token_state = both.x; 
							token_int_cost = both.y; 
							// beam may have changed since generation
							// We are non-emitting in this kernel, using ne offsets
							start = cst_dev_params.d_arc_ne_offsets[token_state];
							int32 end = cst_dev_params.d_arc_ne_offsets[token_state+1];
							degree = end - start;
						}
						int has_valid_nonpruned_token = (start != -1) ? 1 : 0;
						int2 narcs_and_ntokens_prefix_sum = {degree, has_valid_nonpruned_token};
						int2 aggregate, zero2 = {0,0};
						Int2BlockScan(sh_temp_storage_int2_scan).ExclusiveScan(narcs_and_ntokens_prefix_sum, 
								narcs_and_ntokens_prefix_sum,
								zero2,
								PlusPlus(),
								aggregate);
						// Checking if we are not overflowing the main_q
						const int32 total_ntokens = aggregate.y; 
						if((main_q_end + total_ntokens) >= cst_dev_params.q_capacity) {
							lane_counters->q_overflow = 1;
							goto finalize_kernel;
						}
						const int32 degree_prefix_sum = main_q_narcs + narcs_and_ntokens_prefix_sum.x;
						const int32 degree_sum = aggregate.x;
						main_q_narcs += degree_sum;
						if(has_valid_nonpruned_token) {
							const int32 local_main_q_idx = narcs_and_ntokens_prefix_sum.y;
							const int32 main_q_idx = main_q_end + local_main_q_idx;

							cst_dev_params.d_main_q_arc_offsets.channel(ichannel)[main_q_idx] = start;
							cst_dev_params.d_main_q_degrees_prefix_sum.channel(ichannel)[main_q_idx] = degree_prefix_sum;
							cst_dev_params.d_main_q_state_and_cost.channel(ichannel)[main_q_idx] = {token_state,token_int_cost};
							cst_dev_params.d_main_q_info.lane(ilane)[main_q_idx] = cst_dev_params.d_aux_q_info.lane(ilane)[aux_q_idx];
							cst_dev_params.d_main_q_acoustic_cost.lane(ilane)[main_q_idx] = cst_dev_params.d_aux_q_acoustic_cost.lane(ilane)[aux_q_idx]; // TODO remove
						}
						main_q_end += total_ntokens; 
						__syncthreads(); 
					}
					aux_q_end = 0; // aux_q is now considered empty
				}
finalize_kernel:
				if(threadIdx.x == 0) {
					// This main_q is now final for that frame
					lane_counters->main_q_narcs_and_end = {0,main_q_end}; 
					lane_counters->main_q_local_offset = 0;
					// Resetting the number of final tokens in main_q
					// This is just a reset : If we need to read it, we need to call GetBestCost
					lane_counters->nfinals = 0;
				}	
			}
		}

	__global__ void get_best_cost_kernel(DeviceParams cst_dev_params,KernelParams params, bool isfinal, CostType fst_zero) {
		const int nlanes = params.nlanes_used;
		KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
			LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
			const int32 ichannel = params.channel_to_compute[ilane];
			const ChannelCounters *channel_counters = cst_dev_params.d_channels_counters.channel(ichannel);
			const int32 main_q_end = channel_counters->prev_main_q_narcs_and_end.y;
			const int32 global_offset = channel_counters->prev_main_q_global_offset;
			KALDI_CUDA_DECODER_1D_KERNEL_LOOP(idx, main_q_end) {
				const int2 both = cst_dev_params.d_main_q_state_and_cost.channel(ichannel)[idx];
				const int token_state = both.x;
				const int token_int_cost = both.y;
				CostType cost = orderedIntToFloat(token_int_cost);	
				IntegerCostType int_cost = floatToOrderedInt(cost);
				if(isfinal) {
					const CostType final_cost = cst_dev_params.d_fst_final_costs[token_state];
					int_cost = floatToOrderedInt(cost+final_cost);
					if(final_cost != fst_zero) {
						int list_idx = atomicAdd(&lane_counters->nfinals, 1);
						cst_dev_params.d_list_final_tokens_in_main_q.lane(ilane)[list_idx] = {global_offset+idx, int_cost};
					}
				}
				const IntegerCostType cost_as_int = floatToOrderedInt(cost); 
				const int32 global_idx = global_offset+idx;
				const int2 min_and_arg = {int_cost, global_idx}; // sort by cost, put it first
				atomicMinI2(&lane_counters->min_int_cost_and_arg_with_final, min_and_arg); // TODO maybe reduce locally
			}
		}
	}

	template<typename T>
	__global__ void concatenate_lanes_data(DeviceParams cst_dev_params, KernelParams params, LaneMatrixInterface<T> src, T *concat) {
		const int nlanes = params.nlanes_used;
		KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
			int32 beg = params.main_q_end_lane_offsets[ilane];
			int32 end = params.main_q_end_lane_offsets[ilane+1];
			int32 main_q_end = end - beg;
			KALDI_CUDA_DECODER_1D_KERNEL_LOOP(idx, main_q_end) {
				T d = src.lane(ilane)[idx];
				concat[beg+idx] = d;
			}
		}
	}

	__global__ void fill_best_int_cost_kernel(DeviceParams cst_dev_params,KernelParams params) {
		// Operator for the prefix sum inside the CUDA block
		const int nlanes = params.nlanes_used;
		KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
			const LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
			const int32 main_q_end = lane_counters->main_q_narcs_and_end.y;
			KALDI_CUDA_DECODER_1D_KERNEL_LOOP(main_q_idx, main_q_end) {
				// Position of considered token in the main_q
				const int32 ichannel = params.channel_to_compute[ilane];
				if(main_q_idx < main_q_end) {
					int2 both = cst_dev_params.d_main_q_state_and_cost.channel(ichannel)[main_q_idx]; 
					StateId token_state = both.x;
					IntegerCostType token_int_cost = both.y; 
					int local_idx;
					hashmap_insert(cst_dev_params.d_hashmap_values.lane(ilane), token_state, token_int_cost, main_q_idx, cst_dev_params.hashmap_capacity, &local_idx);
					cst_dev_params.d_main_q_n_extra_prev_tokens_local_idx.lane(ilane)[main_q_idx] = local_idx; 
				}
			}
		}	
	}

	__global__ void compute_costs_histogram_kernel(DeviceParams cst_dev_params, KernelParams params, bool use_aux_q) {
		const int nlanes = params.nlanes_used;
		typedef cub::BlockHistogram<BinId,KALDI_CUDA_DECODER_1D_BLOCK,1,KALDI_CUDA_DECODER_NBINS+1> BlockHistogram;
		__shared__ typename BlockHistogram::TempStorage temp_storage;
		__shared__ unsigned int smem_histogram[KALDI_CUDA_DECODER_NBINS+1];

		KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
			const int32 ichannel = params.channel_to_compute[ilane];
			const LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
			const int32 q_end = use_aux_q ? lane_counters->post_expand_aux_q_end : lane_counters->main_q_narcs_and_end.y;
			if(q_end <= cst_dev_params.max_active)
				continue; // nothing to do

			// Reset local histogram for this lane
			BlockHistogram(temp_storage).InitHistogram(smem_histogram);
			CostType beam = orderedIntToFloat(lane_counters->int_beam);
			CostType min_cost = orderedIntToFloat(lane_counters->min_int_cost);
			CostType bin_width = beam / KALDI_CUDA_DECODER_NBINS;

			// We have a sync inside the loop, keeping all threads alive
			KALDI_CUDA_DECODER_1D_BLOCK_OFFSET_KERNEL_LOOP(block_offset, thread_idx, q_end) {
				const int32 q_idx = block_offset + thread_idx;
				// The last bin is for everything we don't want to count:
				// cost already above the beam, or non-valid tokens
				BinId bin_id[1];
				bin_id[0] = KALDI_CUDA_DECODER_NBINS;
				if(q_idx < q_end) {
					IntegerCostType int_cost = use_aux_q
							? cst_dev_params.d_aux_q_state_and_cost.lane(ilane)[q_idx].y
							: cst_dev_params.d_main_q_state_and_cost.channel(ichannel)[q_idx].y;
					CostType cost = orderedIntToFloat(int_cost);
					CostType extra = cost - min_cost;
					// We only count valid tokens
					if(extra < beam) {
						bin_id[0] = (BinId)__fdiv_rd(extra, bin_width);
					}
				}
				BlockHistogram(temp_storage).Composite(bin_id, smem_histogram); // sync
			}
		
			// Not using the macros 1D_LOOP because that loop is only within a CTA	
			for(int32 bin_id_w=threadIdx.x; 
				bin_id_w < KALDI_CUDA_DECODER_NBINS;
				bin_id_w += KALDI_CUDA_DECODER_1D_BLOCK) {
				// Writing the local histo to global
				// We don't care about the last bin (cf above)
				int32 s_count = (int32)smem_histogram[bin_id_w];
				atomicAdd(&cst_dev_params.d_histograms.lane(ilane)[bin_id_w], s_count);
			}
			// Making sure we're done reading from smem
			__syncthreads(); 
		}
	}

	// use only one CTA per lane
	__global__ void update_beam_using_histogram_kernel(DeviceParams cst_dev_params,KernelParams params, bool use_aux_q) {
		typedef cub::BlockScan<int, KALDI_CUDA_DECODER_1D_BLOCK> BlockScan;
		__shared__ typename BlockScan::TempStorage temp_storage;

		const int nlanes = params.nlanes_used;
		const int max_active = cst_dev_params.max_active;
		KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
			LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
			const int32 q_end = use_aux_q ? lane_counters->post_expand_aux_q_end : lane_counters->main_q_narcs_and_end.y;
			if(q_end <= max_active)
				continue; // nothing to do
			CostType beam = orderedIntToFloat(lane_counters->int_beam);
			CostType min_cost = orderedIntToFloat(lane_counters->min_int_cost);
			int32 it_sum = 0;
			// Not using the macros 1D_LOOP because that loop is only within a CTA	
			for(int32 offset=0;
				offset < KALDI_CUDA_DECODER_NBINS;
				offset += KALDI_CUDA_DECODER_1D_BLOCK) {
				int bin_id = offset + threadIdx.x;
				int val = 0;
				if(bin_id < KALDI_CUDA_DECODER_NBINS) {
					val = cst_dev_params.d_histograms.lane(ilane)[bin_id];
					cst_dev_params.d_histograms.lane(ilane)[bin_id] = 0; // reset for next time
				}
				int prefix_sum;
				BlockScan(temp_storage).ExclusiveSum(val, prefix_sum);
				prefix_sum += it_sum; // adding sum from previous for iterations
				//printf("main_histo[%i,%i] = %i (q_end=%i) \n", ilane, bin_id, prefix_sum, q_end);
				if(threadIdx.x == (KALDI_CUDA_DECODER_1D_BLOCK-1))
					it_sum += (prefix_sum+val);

				if(val != 0 && prefix_sum < max_active && (prefix_sum+val) >= max_active) {
					// We found our new beam	
					CostType new_beam = (beam/KALDI_CUDA_DECODER_NBINS)*(bin_id+1);
					//if(use_aux_q) printf("aux:\t");
					//printf("ilane=%i, achieved=%i, max_active=%i, new_beam=%f < %f \n",ilane,(prefix_sum+val), max_active, new_beam, beam);
					IntegerCostType new_int_beam = floatToOrderedInt(new_beam);
					lane_counters->int_beam = new_int_beam; 
					lane_counters->adaptive_int_beam_with_validity_index.x = new_int_beam; 
					lane_counters->int_cutoff = floatToOrderedInt(min_cost + new_beam);
					// keep looping, we need to reset to 0 the histogram
				}
			}

			// Saving our new beam for this lane
		}	
	}

	template __global__ void expand_arcs_kernel<true>(DeviceParams cst_dev_params,KernelParams params);
	template __global__ void expand_arcs_kernel<false>(DeviceParams cst_dev_params,KernelParams params);
	template __global__ void post_expand_kernel<true>(DeviceParams cst_dev_params,KernelParams params);
	template __global__ void post_expand_kernel<false>(DeviceParams cst_dev_params,KernelParams params);
	template __global__ void concatenate_lanes_data<InfoToken>(DeviceParams cst_dev_params,KernelParams params,LaneMatrixInterface<InfoToken> src,InfoToken *concat);
	template __global__ void concatenate_lanes_data<CostType>(DeviceParams cst_dev_params,KernelParams params,LaneMatrixInterface<CostType> src,CostType *concat);
	template __global__ void concatenate_lanes_data<float2>(DeviceParams cst_dev_params,KernelParams params,LaneMatrixInterface<float2> src,float2 *concat);
	template __global__ void concatenate_lanes_data<int32>(DeviceParams cst_dev_params,KernelParams params,LaneMatrixInterface<int32> src,int32 *concat);
} // end namespace kaldi
