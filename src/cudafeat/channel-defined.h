#ifndef CUDAFEAT_CHANNEL_DEFINED_H_
#define CUDAFEAT_CHANNEL_DEFINED_H_

namespace kaldi {

  //definition for channels
  typedef int32_t Channels;
  //channel to store states
  
  template<typename T>
  struct StateCaching {
    T *data;
    int32_t stride;

    __device__ __host__ inline T *lane(const Channels channel) {
      //get pointers
      return &data[stride * channel];
    }
  };

  //kernel params
  struct KernelParam {
    //store channel info;
    Channels compute_channel;
    int32_t start_utt;
    int32_t end_utt;
    int32_t n_frames_valid;
  };

  //device params
  //for mfcc, hold unchanged pointers
  struct DeviceMFCC {
    //store pointer
    StateCaching<float> chunk_channel;
    StateCaching<int32_t> offset_channel;
  };

  //device params
  //for i-vector
  struct DeviceIvector {
    //store pointer
    StateCaching<float> tot_channel;
    StateCaching<float> linear_channel;
    StateCaching<float> quad_channel;
  };

  //device params
  //for cmvn
  //will be stat
  struct DeviceCMVN {
    //store pointers
    StateCaching<float> cmvn_stat;
    StateCaching<int32_t> offset_channel;
    StateCaching<int32_t> buffer_channel;
    StateCaching<float> frame_channel;
  };
}

#endif
