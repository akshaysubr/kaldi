// decoder/cuda-decoder-utils.cu
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


#include "cudadecoder/cuda-decoder-utils.h"
#include <nvToolsExt.h>

namespace kaldi {

	/***************************************CudaFst Implementation*****************************************/

	void CudaFst::Initialize(const fst::Fst<StdArc> &fst, const TransitionModel &trans_model) {
		nvtxRangePushA("CudaFst constructor");
		//count states since Fst doesn't provide this functionality
		num_states_=0;
		for( fst::StateIterator<fst::Fst<StdArc> > iter(fst); !iter.Done(); iter.Next()) 
			++num_states_;
		
		start_=fst.Start();

		//allocate and initialize offset arrays
		h_final_.resize(num_states_);
		h_e_offsets_.resize(num_states_+1);
		h_ne_offsets_.resize(num_states_+1);
    
    d_e_offsets_=static_cast<unsigned int*>(CuDevice::Instantiate().Malloc((num_states_+1)*sizeof(*d_e_offsets_)));
    d_ne_offsets_=static_cast<unsigned int*>(CuDevice::Instantiate().Malloc((num_states_+1)*sizeof(*d_ne_offsets_)));
    d_final_=static_cast<float*>(CuDevice::Instantiate().Malloc((num_states_)*sizeof(*d_final_)));
		KALDI_ASSERT(d_e_offsets_);
		KALDI_ASSERT(d_ne_offsets_);
		KALDI_ASSERT(d_final_);
  	
		//iterate through states and arcs and count number of arcs per state
		e_count_=0;
		ne_count_=0;

		// Init first offsets
		h_ne_offsets_[0] = 0; 
		h_e_offsets_[0] = 0; 

		for(int i=0;i<num_states_;i++) {
			h_final_[i]=fst.Final(i).Value();
			//count emiting and non_emitting arcs
			for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, i); !aiter.Done(); aiter.Next()) {
				StdArc arc = aiter.Value();
				int32 ilabel = arc.ilabel;
				if(ilabel!=0) { //emitting
					e_count_++;
				} else { //non-emitting
					ne_count_++;
				}
			}
			h_ne_offsets_[i+1] = ne_count_;
			h_e_offsets_[i+1] = e_count_;
		}

		// We put the emitting arcs before the nonemitting arcs in the arc list
		// adding offset to the non emitting arcs
		// we go to num_states_+1 to take into account the last offset
		for(int i=0;i<num_states_+1;i++) 
			h_ne_offsets_[i]+=e_count_;   //e_arcs before

		arc_count_=e_count_+ne_count_;

		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpy(d_e_offsets_,&h_e_offsets_[0],(num_states_+1)*sizeof(*d_e_offsets_),cudaMemcpyHostToDevice));
		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpy(d_ne_offsets_,&h_ne_offsets_[0],(num_states_+1)*sizeof(*d_ne_offsets_),cudaMemcpyHostToDevice));
		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpy(d_final_,&h_final_[0],num_states_*sizeof(*d_final_),cudaMemcpyHostToDevice));

		h_arc_weights_.resize(arc_count_);
		h_arc_nextstates_.resize(arc_count_);
		// ilabels (nnet3 indexing)
		std::vector<int32> h_arc_pdf_ilabels_(arc_count_);
		// ilabels (fst indexing)
		h_arc_id_ilabels_.resize(arc_count_);
		h_arc_olabels_.resize(arc_count_);

		d_arc_weights_=static_cast<float*>(CuDevice::Instantiate().Malloc(arc_count_*sizeof(*d_arc_weights_)));
    d_arc_nextstates_=static_cast<StateId*>(CuDevice::Instantiate().Malloc(arc_count_*sizeof(*d_arc_nextstates_)));

    // Only the ilabels for the e_arc are needed on the device
    d_arc_pdf_ilabels_=static_cast<int32*>(CuDevice::Instantiate().Malloc(e_count_*sizeof(*d_arc_pdf_ilabels_)));

    KALDI_ASSERT(d_arc_weights_);
    KALDI_ASSERT(d_arc_nextstates_);
    KALDI_ASSERT(d_arc_pdf_ilabels_);

		// We do not need the olabels on the device - GetBestPath is on CPU

		//now populate arc data
		int e_idx=0;
		int ne_idx=e_count_; //starts where e_offsets_ ends
		for(int i=0;i<num_states_;i++) {
			for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, i); !aiter.Done(); aiter.Next()) {
				StdArc arc = aiter.Value();
				int idx;
				if(arc.ilabel!=0) { //emitting
					idx=e_idx++;
				} else {
					idx=ne_idx++;
				}
				h_arc_weights_[idx] = arc.weight.Value();
				h_arc_nextstates_[idx] = arc.nextstate;
  				// Converting ilabel here, to avoid reindexing when reading nnet3 output
				h_arc_id_ilabels_[idx]=arc.ilabel;
				int32 ilabel_pdf = trans_model.TransitionIdToPdf(arc.ilabel);
				h_arc_pdf_ilabels_[idx]=ilabel_pdf;
				h_arc_olabels_[idx]=arc.olabel;
			}
		}

		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpy(d_arc_weights_,&h_arc_weights_[0],arc_count_*sizeof(*d_arc_weights_),cudaMemcpyHostToDevice));
		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpy(d_arc_nextstates_,&h_arc_nextstates_[0],arc_count_*sizeof(*d_arc_nextstates_),cudaMemcpyHostToDevice));
		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpy(d_arc_pdf_ilabels_,&h_arc_pdf_ilabels_[0],e_count_*sizeof(*d_arc_pdf_ilabels_),cudaMemcpyHostToDevice));
		
		// Making sure the graph is ready
		cudaDeviceSynchronize();
		KALDI_DECODER_CUDA_CHECK_ERROR();
		nvtxRangePop();
	}

	void CudaFst::Finalize() {
		nvtxRangePushA("CudaFst destructor");
    CuDevice::Instantiate().Free(d_e_offsets_);
    CuDevice::Instantiate().Free(d_ne_offsets_);
    CuDevice::Instantiate().Free(d_final_);
    CuDevice::Instantiate().Free(d_arc_weights_);
    CuDevice::Instantiate().Free(d_arc_nextstates_);
    CuDevice::Instantiate().Free(d_arc_pdf_ilabels_);
    nvtxRangePop();
	}


	/***************************************End CudaFst****************************************************/


	// Constructor always takes an initial capacity for the vector
	// even if the vector can grow if necessary, it damages performance
	// we need to have an appropriate initial capacity (is set using a parameter in CudaDecoderConfig)
	InfoTokenVector::InfoTokenVector(int32 capacity, cudaStream_t copy_st) : capacity_(capacity), copy_st_(copy_st) {
		KALDI_VLOG(2) << "Allocating InfoTokenVector with capacity = " << capacity_ << " tokens";
		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMallocHost(&h_data_, capacity_ * sizeof(*h_data_))); 
		Reset();
	}

        InfoTokenVector::InfoTokenVector(const InfoTokenVector &other) : InfoTokenVector(other.capacity_, other.copy_st_) {}

	void InfoTokenVector::Reset() {
		size_ = 0;
	};

	void InfoTokenVector::CopyFromDevice(InfoToken *d_ptr, int32 count) { // TODO add the Append keyword 
		Reserve(size_+count); // making sure we have the space

		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpyAsync(&h_data_[size_], d_ptr, count*sizeof(*h_data_), cudaMemcpyDeviceToHost, copy_st_));
		size_ += count;
	}

	void InfoTokenVector::Clone(const InfoTokenVector &other) {
		Reserve(other.Size());
		size_ = other.Size();
		if(size_ == 0)
			return;
		const InfoToken *h_data_other = other.GetRawPointer();
		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpyAsync(h_data_, h_data_other, size_ * sizeof(*h_data_), cudaMemcpyHostToHost, copy_st_));
		cudaStreamSynchronize(copy_st_); // after host2host?
	};

	void InfoTokenVector::Reserve(int32 min_capacity) {
		if(min_capacity <= capacity_)
			return;

		while(capacity_ < min_capacity)
			capacity_ *= 2;

		KALDI_VLOG(2) << "Reallocating InfoTokenVector on host (new capacity = " << capacity_ << " tokens).";

		cudaStreamSynchronize(copy_st_);
		InfoToken *h_old_data = h_data_;
		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMallocHost(&h_data_, capacity_ * sizeof(*h_data_))); 

		if(!h_data_)
			KALDI_ERR << "Host ran out of memory to store tokens. Exiting.";

		if(size_ > 0)
			KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpyAsync(h_data_, h_old_data, size_ * sizeof(*h_data_), cudaMemcpyHostToHost, copy_st_));

		cudaStreamSynchronize(copy_st_);
		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaFreeHost(h_old_data));
	}

	InfoToken * InfoTokenVector::GetRawPointer() const {
		return h_data_;
	}

	InfoTokenVector::~InfoTokenVector() {
		KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaFreeHost(h_data_));
	}

} // end namespace kaldi
