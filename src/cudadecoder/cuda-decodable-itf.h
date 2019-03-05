// cudadecoder/cuda-decodable-itf.h
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

#ifndef KALDI_DECODABLE_ITF_H
#define KALDI_DECODABLE_ITF_H

#include "itf/decodable-itf.h"

namespace kaldi {

class CudaDecodableInterface : public DecodableInterface {
public:
  virtual BaseFloat *GetLogLikelihoodsCudaPointer(int32 subsampled_frame) = 0;
};
}

#endif
