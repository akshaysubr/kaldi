// cudadecoder/cuda-decoder.h
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

#ifndef KALDI_DECODER_CUDA_DECODER_H_
#define KALDI_DECODER_CUDA_DECODER_H_

#include <cuda_runtime_api.h>
#include <tuple>
#include <vector>

#include "cudadecoder/cuda-decodable-itf.h"
#include "cudadecoder/cuda-decoder-utils.h"
#include "cudadecoder/cuda-fst.h"
#include "nnet3/decodable-online-looped.h"
#include "util/stl-utils.h"

// A decoder channel is linked to one utterance. Frames
// from the same must be sent to the same channel.
//
// A decoder lane is where the computation actually happens
// a decoder lane is given a frame and its associated channel
// and does the actual computation
//
// An analogy would be lane -> a core, channel -> a software thread

// Number of GPU decoder lanes
#define KALDI_CUDA_DECODER_MAX_N_LANES 200

// If we're at risk of filling the tokens queue,
// the beam is reduced to keep only the best candidates in the
// remaining space
// We then slowly put the beam back to its default value
// beam_next_frame = min(default_beam, RECOVER_RATE * beam_previous_frame)
#define KALDI_CUDA_DECODER_ADAPTIVE_BEAM_RECOVER_RATE 1.2f

// Defines for the cuda decoder kernels
// It shouldn't be necessary to change the DIMX of the kernels

// Below that value, we launch the persistent kernel for NonEmitting
#define KALDI_CUDA_DECODER_NONEM_LT_MAX_NARCS 4096

// We know we will have at least X elements in the hashmap
// We allocate space for X*KALDI_CUDA_DECODER_HASHMAP_CAPACITY_FACTOR elements
// to avoid having too much collisions
#define KALDI_CUDA_DECODER_HASHMAP_CAPACITY_FACTOR 1

// Max size of the total kernel arguments
// 4kb for compute capability >= 2.0
#define KALDI_CUDA_DECODER_MAX_KERNEL_ARGUMENTS_BYTE_SIZE (4096)

// When applying the max-active, we need to compute a topk
// to perform that (soft) topk, we compute a histogram
// here we define the number of bins in that histogram
// it has to be less than the number of 1D threads 
#define KALDI_CUDA_DECODER_HISTO_NBINS 255

// Adaptive beam parameters 
// We will decrease the beam when we detect that we are generating too many tokens
// for the first segment of the aux_q, we don't do anything (keep the original beam)
// the first segment is made of (aux_q capacity)/KALDI_CUDA_DECODER_ADAPTIVE_BEAM_STATIC_SEGMENT
// then we will decrease the beam step by step, until 0.
// we will decrease the beam every m elements, with:
// x = (aux_q capacity)/KALDI_CUDA_DECODER_ADAPTIVE_BEAM_STATIC_SEGMENT (static segment
// y = (aux_q capacity) - x
// m = y / KALDI_CUDA_DECODER_ADAPTIVE_BEAM_NBINS
#define KALDI_CUDA_DECODER_ADAPTIVE_BEAM_STATIC_SEGMENT 4
#define KALDI_CUDA_DECODER_ADAPTIVE_BEAM_NBINS 8

namespace kaldi {
namespace CudaDecode {
enum OVERFLOW_TYPE {
  OVERFLOW_NONE = 0,
  OVERFLOW_MAIN_Q = 1,
  OVERFLOW_AUX_Q = 2
};

// Forward declaration 
class DeviceParams;
class KernelParams;
class HashmapValueT; // TODO why forward?

class CudaDecoder;

struct CudaDecoderConfig {
  BaseFloat default_beam;
  BaseFloat lattice_beam;
  int32 max_tokens;
  int32 max_tokens_per_frame;
  int32 nlanes;
  int32 nchannels;
  int32 max_active;

  CudaDecoderConfig()
      : default_beam(15.0),
        lattice_beam(10.0),
        max_tokens(2000000),
        max_tokens_per_frame(1000000),
        max_active(10000) {}

  void Register(OptionsItf *opts) {
    opts->Register(
        "beam", &default_beam,
        "Decoding beam.  Larger->slower, more accurate. The beam may be"
        "decreased if we are generating too many tokens compared to "
        "what the queue can hold (max_tokens_per_frame)");
    opts->Register("max-tokens-pre-allocated", &max_tokens,
                   "Total number of tokens pre-allocated (equivalent to "
                   "reserve in a std vector).  If actual usaged exceeds this "
                   "performance will be degraded");
    opts->Register("max-tokens-per-frame", &max_tokens_per_frame,
                   "Number of tokens allocated per frame. If actual usaged "
                   "exceeds this the results are undefined.");
    opts->Register("lattice-beam", &lattice_beam, "Lattice generation beam");
    opts->Register("max-active", &max_active,
                   "Max number of tokens active for each frame");
  }
  void Check() const {
    KALDI_ASSERT(default_beam > 0.0 && max_tokens > 0 &&
                 max_tokens_per_frame > 0 && lattice_beam >= 0 &&
                 max_active > 1);
  }
};

class CudaDecoder {
 public:
  // Creating a new CudaDecoder, associated to the FST fst
  // nlanes and nchannels are defined as follow

  // A decoder channel is linked to one utterance.
  // When we need to perform decoding on an utterance,
  // we pick an available channel, call InitDecoding on that channel
  // (with that ChannelId in the channels vector in the arguments)
  // then call AdvanceDecoding whenever frames are ready for the decoder
  // for that utterance (also passing the same ChannelId to AdvanceDecoding)
  //
  // A decoder lane is where the computation actually happens
  // a decoder lane is channel, and perform the actual decoding
  // of that channel.
  // If we have 200 lanes, we can compute 200 utterances (channels)
  // at the same time. We need many lanes in parallel to saturate the big GPUs
  //
  // An analogy would be lane -> a CPU core, channel -> a software thread
  // A channel saves the current state of the decoding for a given utterance.
  // It can be kept idle until more frames are ready to be processed
  //
  // We will use as many lanes as necessary to saturate the GPU, but not more.
  // A lane has an higher memory usage than a channel. If you just want to be
  // able to
  // keep more audio channels open at the same time (when I/O is the bottleneck
  // for instance,
  // typically in the context of online decoding), you should instead use more
  // channels.
  //
  // A channel is typically way smaller in term of memory usage, and can be used
  // to oversubsribe lanes in the context of online decoding
  // For instance, we could choose nlanes=200 because it gives us good
  // performance
  // on a given GPU. It gives us an end-to-end performance of 3000 XRTF. We are
  // doing online,
  // so we only get audio at realtime speed for a given utterance/channel.
  // We then decide to receive audio from 2500 audio channels at the same time
  // (each at realtime speed),
  // and as soon as we have frames ready for nlanes=200 channels, we call
  // AdvanceDecoding on those channels
  // In that configuration, we have nlanes=200 (for performance), and
  // nchannels=2500 (to have enough audio
  // available at a given time).
  // Using nlanes=2500 in that configuration would first not be possible (out of
  // memory), but also not necessary.
  // Increasing the number of lanes is only useful if it increases performance.
  // If the GPU is saturated at nlanes=200,
  // you should not increase that number
  CudaDecoder(const CudaFst &fst, const CudaDecoderConfig &config, int32 nlanes,
              int32 nchannels);
  ~CudaDecoder();

  // InitDecoding initializes the decoding, and should only be used if you
  // intend to call AdvanceDecoding() on the channels listed in channels
  void InitDecoding(const std::vector<ChannelId> &channels);

  // AdvanceDecoding on a given batch
  // a batch is defined by the channels vector
  // We can compute N channels at the same time (in the same batch)
  // where N = number of lanes, as defined in the constructor 
  // AdvanceDecoding will compute as many frames as possible while running the full batch
  // when at least one channel has no more frames ready to be computed, AdvanceDecoding returns
  // The user then decides what to do, i.e.:
  // 
  // 1) either remove the empty channel from the channels list
  // and call again AdvanceDecoding
  // 2) or swap the empty channel with another one that has frames ready
  // and call again AdvanceDecoding
  //
  // Solution 2) should be preferred because we need to run full, big batches to
  // saturate the GPU
  //
  // If max_num_frames is >= 0 it will decode no more than
  // that many frames.
  void AdvanceDecoding(const std::vector<ChannelId> &channels,
                       std::vector<CudaDecodableInterface *> &decodables,
                       int32 max_num_frames = -1);

  // Returns the number of frames already decoded in a given channel
  int32 NumFramesDecoded(ChannelId ichannel) const;

  // GetBestPath gets the one-best decoding traceback. If "use_final_probs" is
  // true
  // AND we reached a final state, it limits itself to final states;
  // otherwise it gets the most likely token not taking into account
  // final-probs.
  void GetBestPath(const std::vector<ChannelId> &channels,
                   std::vector<Lattice *> &fst_out_vec,
                   bool use_final_probs = true);
  // GetRawLattice gets the lattice decoding traceback (using the lattice-beam
  // in the CudaConfig parameters).
  // If "use_final_probs" is true
  // AND we reached a final state, it limits itself to final states;
  // otherwise it gets the most likely token not taking into account
  // final-probs.
  void GetRawLattice(const std::vector<ChannelId> &channels,
                     std::vector<Lattice *> &fst_out_vec, bool use_final_probs);

  // GetBestCost finds the best cost in the last tokens queue
  // for each channel in channels. If isfinal is true,
  // we also add the final cost to the token costs before
  // finding the minimum cost
  // We list all tokens that have a cost within [best; best+lattice_beam]
  // in list_lattice_tokens.
  // We alsos set has_reached_final[ichannel] to true if token associated to a
  // final state
  // exists in the last token queue of that channel
  void GetBestCost(
      const std::vector<ChannelId> &channels, bool isfinal,
      std::vector<std::pair<int32, CostType>> *argmins,
      std::vector<std::vector<std::pair<int, float>>> *list_lattice_tokens,
      std::vector<bool> *has_reached_final);

 private:
  void AllocateDeviceData();
  void AllocateHostData();
  void AllocateDeviceKernelParams();
  void InitDeviceData();
  void InitHostData();
  void InitDeviceParams();

  // Computes the initial channel
  // The initial channel is used to initialize a channel
  // when a new utterance starts (we clone it into the given channel)
  void ComputeInitialChannel();

  // Updates *h_kernel_params using channels
  void SetChannelsInKernelParams(const std::vector<ChannelId> &channels);

  // Context-switch functions
  // Used to perform the context-switch of load/saving the state of a channels
  // into a lane. When a channel will be executed on a lane, we load that
  // channel
  // into that lane (same idea than when we load a software threads into a CPU
  // core registers)
  void LoadChannelsStateToLanes();
  void SaveChannelsStateFromLanes();

  // If we have more than max_active_ tokens in the queue, we will compute a new
  // beam,
  // that will only keep max_active_ tokens
  void ApplyMaxActiveAndReduceBeam(bool use_aux_q);

  // TODO comments
  void ExpandArcsEmitting();

  // CheckOverflow
  // If a kernel sets the flag h_q_overflow, we send a warning to stderr
  // Overflows are detected and prevented on the device. It only means
  // that we've discarded the tokens that were created after the queue was full
  // That's why we only send a warning. It is not a fatal error
  void CheckOverflow();

  // Evaluates the function func for each lane, returning the max of all return
  // values
  // (func returns int32)
  // Used for instance to ge the max number of arcs for all lanes
  int32 GetMaxForAllLanes(std::function<int32(const LaneCounters &)> func);
  int32 NumFramesToDecode(const std::vector<ChannelId> &channels,
                          std::vector<CudaDecodableInterface *> &decodables,
                          int32 max_num_frames);
  // Copy the lane counters back to host, async, using stream st
  // The lanes counters contain all the information such as main_q_end (number
  // of tokens in the main_q)
  // main_q_narcs (number of arcs) during the computation. That's why we
  // frequently copy it back to host
  // to know what to do next
  void CopyLaneCountersToHostAsync(cudaStream_t st);

  // The selected tokens for each frame will be copied back to host. We will
  // store them
  // on host memory, and we wil use them to create the final lattice once we've
  // reached the last frame
  // We will also copy information on those tokens that we've generated on the
  // device, such as
  // which tokens are associated to the same FST state in the same frame, or
  // their extra cost.
  // We cannot call individuals Device2Host copies for each channel, because it
  // would lead to a lot of
  // small copies, reducing performance. Instead we concatenate all channels
  // data into a single
  // continuous array, copy that array to host, then unpack it to the individual
  // channel vectors
  // The first step (pack then copy to host, async) is done in
  // PerformConcatenatedCopy
  // The second step is done in MoveConcatenatedCopyToVector
  // A sync on cudaStream st has to happen between the two functions to make
  // sure that the copy is done
  //
  // Each lane contains X elements to be copied, where X = func(ilane)
  // That data is contained in the array (pointer, X), with pointer = src[ilane]
  // It will be concatenated in d_concat on device, then copied async into
  // h_concat
  // That copy is launched on st
  // The offset of the data of each lane in the concatenate array is saved in
  // *lanes_offsets_ptr
  // it will be used for unpacking in MoveConcatenatedCopyToVector
  template <typename T>
  void PerformConcatenatedCopy(std::function<int32(const LaneCounters &)> func,
                               LaneMatrixInterface<T> src, T *d_concat,
                               T *h_concat, cudaStream_t st,
                               std::vector<int32> *lanes_offsets_ptr);
  template <typename T>
  void MoveConcatenatedCopyToVector(const std::vector<int32> &lanes_offsets,
                                    T *h_concat,
                                    std::vector<std::vector<T>> *vecvec);

  // Computes a set of static asserts on the static values
  // such as the defines : KALDI_CUDA_DECODER_MAX_N_LANES for example
  // In theory we should do them at compile time
  void CheckStaticAsserts();

  // Data members

  // The CudaFst data structure contains the FST graph
  // in the CSR format
  const CudaFst fst_;

  // Counters used by a decoder lane
  // Contains all the single values generated during computation,
  // such as the current size of the main_q, the number of arcs currently in
  // that queue
  // We load data from the channel state during context-switch (for instance the
  // size of the last token queue for that channel)
  LaneCounters *h_lanes_counters_;
  // Counters of channels
  // Contains all the single values saved to remember the state of a channel
  // not used during computation. Those values are loaded/saved into/from a lane
  // during context switching
  ChannelCounters *h_channels_counters_;
  // Contain the various counters used by lanes/channels, such as main_q_end,
  // main_q_narcs. On device memory (equivalent of h_channels_counters on
  // device)
  DeviceChannelMatrix<ChannelCounters> d_channels_counters_;
  DeviceLaneMatrix<LaneCounters> d_lanes_counters_;
  // Number of lanes and channels, as defined in the constructor arguments
  int32 nlanes_, nchannels_;

  //
  // We will now define the data used on the GPU
  // The data is mainly linked to two token queues
  // - the main queue
  // - the auxiliary queue
  //
  // The auxiliary queue is used to store the raw output of ExpandArcs.
  // We then prune that aux queue (and apply max-active) and move the survival
  // tokens in the main
  // queue.
  // Tokens stored in the main q can then be used to generate new tokens (using
  // ExpandArcs)
  // We also generate more information about what's in the main_q at the end of
  // a frame
  // such as which tokens are associated to the same FST state
  //
  // As a reminder, here's the data structure of a token :
  //
  // struct Token { state, cost, prev_token, arc_idx }
  //
  // Please keep in mind that this structure is also used in the context
  // of lattice decoding. We are not storing a list of forward links like in the
  // CPU
  // decoder. A token stays an instanciation of an single arc.
  //
  // For performance reasons, we split the tokens in three parts :
  // { state } , { cost }, { prev_token, arc_idx }
  // Each part has its associated queue
  // For instance, d_main_q_state[i], d_main_q_cost[i], d_main_q_info[i]
  // all refer to the same token (at index i)
  // The data structure InfoToken contains { prev_token, arc_idx }
  //
  // main_q
  DeviceChannelMatrix<int2> d_main_q_state_and_cost_;
  // Usually contains {prev_token, arc_idx}
  // If more than one token is associated to a fst_state,
  // it will contain where to find the list of those tokens in
  // d_main_q_extra_prev_tokens
  // ie {offset,size} in that list. We differentiate the two situations by
  // calling InfoToken.IsUniqueTokenForStateAndFrame()
  DeviceLaneMatrix<InfoToken> d_main_q_info_;
  // Acoustic cost of a given token
  DeviceLaneMatrix<CostType> d_main_q_acoustic_cost_;
  // Prefix sum of the arc's degrees in the main_q. Used by expand_arcs
  DeviceChannelMatrix<int32> d_main_q_degrees_prefix_sum_;
  // d_main_q_arc_offsets[i] = fst_.arc_offsets[d_main_q_state[i]]
  // we pay the price for the random memory accesses of fst_.arc_offsets in the
  // preprocess kernel
  // we cache the results in d_main_q_arc_offsets which will be read in a
  // coalesced fashion in expand
  DeviceChannelMatrix<int32> d_main_q_arc_offsets_;

  // At the end of a frame, we use a hashmap to detect the tokens that are
  // associated with the same FST state S
  // We do it that the very end, to only use the hashmap on post-prune, post-max
  // active tokens
  DeviceLaneMatrix<HashmapValueT> d_hashmap_values_;
  // When more than one token is associated to a single FST state,
  // we will list those tokens into another list : d_main_q_extra_prev_tokens
  // we will also save data useful is such a case, such as the extra_cost of a
  // token compared
  // to the best for that state
  DeviceLaneMatrix<InfoToken> d_main_q_extra_prev_tokens_;
  DeviceLaneMatrix<float2> d_main_q_extra_cost_;  // TODO rename because it also
                                                  // contains the acoustic_cost

  // Histogram. Used to perform the histogram of the token costs
  // in the main_q. Used to perform a soft topk of the main_q (max-active)
  DeviceLaneMatrix<int32> d_histograms_;

  // Used when generating d_main_q_extra_prev_tokens
  // This happens when we have multiple tokens for a given state S
  // The best token for S (always unique, even if multiple tokens have the best
  // cost)
  // will be the representative of that state S. d_main_q_representative_id_
  // contains the id
  // of their representative for each token.
  // If only one token is associated with S, its representative will be itself
  DeviceLaneMatrix<int32> d_main_q_representative_id_;
  // local_idx of the extra cost list for a state
  // For a given state S, first token associated with S will have local_idx=0
  // the second one local_idx=1, etc. The order of the local_idxs is random
  DeviceLaneMatrix<int32> d_main_q_n_extra_prev_tokens_local_idx_;
  // Where to write the extra_prev_tokens in the d_main_q_extra_prev_tokens_
  // queue
  DeviceLaneMatrix<int32> d_main_q_extra_prev_tokens_prefix_sum_;

  // Used when computing the prefix_sums in preprocess_in_place. Stores
  // the local_sums per CTA
  DeviceLaneMatrix<int2> d_main_q_block_sums_prefix_sum_;

  // Defining the aux_q
  DeviceLaneMatrix<int2> d_aux_q_state_and_cost_;
  DeviceLaneMatrix<CostType> d_aux_q_acoustic_cost_;
  DeviceLaneMatrix<InfoToken> d_aux_q_info_;

  // Parameters used by the kernels
  // DeviceParams contains all the parameters that won't change
  // i.e. memory address of the main_q for instance
  // KernelParams contains information that can change.
  // For instance which channel is executing on which lane
  DeviceParams *h_device_params_;
  KernelParams *h_kernel_params_;

  // Initial lane
  // When starting a new utterance,
  // init_channel_id is used to initialize a channel
  int32 init_channel_id_;

  // CUDA streams used by the decoder
  cudaStream_t compute_st_;

  // Parameters extracted from CudaDecoderConfig
  // Those are defined in CudaDecoderConfig
  CostType default_beam_;
  CostType lattice_beam_;
  int32 max_tokens_;
  int32 max_active_;
  int32 max_tokens_per_frame_;
  // Hashmap capacity. Multiple of max_tokens_per_frame
  int32 hashmap_capacity_;

  // Keep track of the number of frames decoded in the current file.
  std::vector<int32> num_frames_decoded_;
  // Offsets of each frame in h_all_tokens_info_
  // for instance, frame 4 of channel 2 has an offset of frame_offsets[2][4]
  std::vector<std::vector<int32>> frame_offsets_;
  std::vector<int32> main_q_emitting_end_;

  // Data storage. We store on host what we will need in
  // GetRawLattice/GetBestPath
  std::vector<std::vector<InfoToken>> h_all_tokens_info_;
  std::vector<std::vector<CostType>> h_all_tokens_acoustic_cost_;
  std::vector<std::vector<InfoToken>> h_all_tokens_extra_prev_tokens_;
  std::vector<std::vector<float2>> h_all_tokens_extra_prev_tokens_extra_cost_;

  // Pinned memory arrays. Used for the DeviceToHost copies
  float2 *h_extra_cost_concat_, *d_extra_cost_concat_;
  InfoToken *h_infotoken_concat_, *d_infotoken_concat_;
  CostType *h_acoustic_cost_concat_, *d_acoustic_cost_concat_;
  InfoToken *h_extra_prev_tokens_concat_;

  // Offsets used in MoveConcatenatedCopyToVector
  std::vector<int32> h_main_q_end_lane_offsets_,
      h_emitting_main_q_end_lane_offsets_;
  std::vector<int32> h_n_extra_prev_tokens_lane_offsets_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(CudaDecoder);
};

}  // end namespace CudaDecode
}  // end namespace kaldi

#endif
