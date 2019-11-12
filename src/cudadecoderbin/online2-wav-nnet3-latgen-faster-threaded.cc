// online2bin/online2-wav-nnet3-latgen-faster-threaded.cc

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

#include <sstream>
#include "cudadecoder/thread-pool.h"
#include "feat/wave-reader.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "nnet3/nnet-utils.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-timing.h"
#include "online2/onlinebin-util.h"
#include "util/kaldi-thread.h"

namespace kaldi {

std::mutex stdout_mutex;

void GetDiagnosticsAndPrintOutput(const std::string &utt,
                                  const fst::SymbolTable *word_syms,
                                  const CompactLattice &clat,
                                  int64 &tot_num_frames, double &tot_like) {
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
  {
    std::lock_guard<std::mutex> lk(stdout_mutex);
    tot_num_frames += num_frames;
    tot_like += likelihood;
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
}
}  // namespace kaldi

using namespace kaldi;
using namespace fst;

struct DecodeParams {
  std::string utt;
  std::string key;
  WaveData *wave_data;
  CompactLatticeWriter *clat_writer;
  const fst::Fst<fst::StdArc> *decode_fst;
  const nnet3::DecodableNnetSimpleLoopedInfo *decodable_info;
  const TransitionModel *trans_model;
  const OnlineNnet2FeaturePipelineInfo *feature_info;
  const LatticeFasterDecoderConfig *decoder_opts;
  const fst::SymbolTable *word_syms;
  int64 *num_frames;
  double *tot_like;
  std::mutex *clat_write_mutex;
  bool write_lattice;
};

void ProcessOneWave(DecodeParams p) {
  // get the data for channel zero (if the signal is not mono, we only
  // take the first channel).
  SubVector<BaseFloat> data(p.wave_data->Data(), 0);

  OnlineNnet2FeaturePipeline feature_pipeline(*p.feature_info);

  SingleUtteranceNnet3Decoder decoder(*p.decoder_opts, *p.trans_model,
                                      *p.decodable_info, *p.decode_fst,
                                      &feature_pipeline);

  BaseFloat samp_freq = p.wave_data->SampFreq();

  int32 samp_offset = 0;
  int32 num_samp = data.Dim();

  SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
  feature_pipeline.AcceptWaveform(samp_freq, wave_part);
  feature_pipeline.InputFinished();

  decoder.AdvanceDecoding();
  decoder.FinalizeDecoding();

  CompactLattice clat;
  bool end_of_utterance = true;
  decoder.GetLattice(end_of_utterance, &clat);

  GetDiagnosticsAndPrintOutput(p.utt, p.word_syms, clat, *p.num_frames,
                               *p.tot_like);

  if (p.write_lattice) {
    std::lock_guard<std::mutex> lk(*p.clat_write_mutex);
    p.clat_writer->Write(p.key, clat);
  }
  // KALDI_LOG << "Decoded utterance " << utt;

  delete p.wave_data;
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
        "optional endpointing.  Note: some configuration values and inputs "
        "are\n"
        "set via config files whose filenames are passed as options\n"
        "\n"
        "Usage: online2-wav-nnet3-latgen-faster [options] <nnet3-in> <fst-in> "
        "<spk2utt-rspecifier> <wav-rspecifier> <lattice-wspecifier>\n"
        "The spk2utt-rspecifier can just be <utterance-id> <utterance-id> if\n"
        "you want to decode utterance by utterance.\n";

    ParseOptions po(usage);

    std::string word_syms_rxfilename;

    // feature_opts includes configuration for the iVector adaptation,
    // as well as the basic features.
    OnlineNnet2FeaturePipelineConfig feature_opts;
    nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
    LatticeFasterDecoderConfig decoder_opts;

    int cur = 0;
    bool write_lattice = true;
    int num_todo = -1;
    int iterations = 1;
    int num_threads = 1;

    po.Register("write-lattice", &write_lattice,
                "Output lattice to a file. Setting to false is useful when "
                "benchmarking");
    po.Register("iterations", &iterations,
                "Number of times to decode the corpus.");
    po.Register("file-limit", &num_todo,
                "Limits the number of files that are processed by this driver. "
                "After N files are processed the remaining files are ignored. "
                "Useful for profiling");
    po.Register("num-threads", &num_threads, "number of threads in workpool.");
    po.Register("word-symbol-table", &word_syms_rxfilename,
                "Symbol table for words [for debug output]");
    po.Register("num-threads-startup", &g_num_threads,
                "Number of threads used when initializing iVector extractor.");

    feature_opts.Register(&po);
    decodable_opts.Register(&po);
    decoder_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      return 1;
    }

    kaldi::cuda_decoder::ThreadPool work_pool(num_threads);

    std::string nnet3_rxfilename = po.GetArg(1), fst_rxfilename = po.GetArg(2),
                spk2utt_rspecifier = po.GetArg(3),
                wav_rspecifier = po.GetArg(4), clat_wspecifier = po.GetArg(5);

    OnlineNnet2FeaturePipelineInfo feature_info(feature_opts);
    feature_info.ivector_extractor_info.use_most_recent_ivector = true;
    feature_info.ivector_extractor_info.greedy_ivector_extractor = true;

    Matrix<double> global_cmvn_stats;
    if (feature_info.global_cmvn_stats_rxfilename != "")
      ReadKaldiObject(feature_info.global_cmvn_stats_rxfilename,
                      &global_cmvn_stats);

    TransitionModel trans_model;
    nnet3::AmNnetSimple am_nnet;
    {
      bool binary;
      Input ki(nnet3_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
      SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
      SetDropoutTestMode(true, &(am_nnet.GetNnet()));
      nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));
    }

    // this object contains precomputed stuff that is used by all decodable
    // objects.  It takes a pointer to am_nnet because if it has iVectors it has
    // to modify the nnet to accept iVectors at intervals.
    nnet3::DecodableNnetSimpleLoopedInfo decodable_info(decodable_opts,
                                                        &am_nnet);

    fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldiGeneric(fst_rxfilename);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_rxfilename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_rxfilename;

    int32 num_task_submitted = 0, num_err = 0;
    int64 num_frames = 0;
    double tot_like = 0.0;
    double total_audio = 0;

    CompactLatticeWriter clat_writer;
    std::mutex clat_write_mutex;

    clat_writer.Open(clat_wspecifier);

    std::vector<std::future<void> > futures;
    Timer timer;

    for (int iter = 0; iter < iterations; iter++) {
      SequentialTableReader<WaveHolder> wav_reader_new(wav_rspecifier);
      for (; !wav_reader_new.Done(); wav_reader_new.Next()) {
        std::string utt = wav_reader_new.Key();

        // Deep copy for now,  Dan mentioned we might be able to use Swap
        // instead.
        WaveData *wave_data = new WaveData(wav_reader_new.Value());

        if (iter == 0) {
          total_audio += wave_data->Duration();
        }

        DecodeParams p;

        p.utt = utt;
        p.key = utt;
        if (iterations > 0) {
          p.key = std::to_string(iter) + "-" + utt;
        }
        p.wave_data = wave_data;
        p.clat_writer = &clat_writer;
        p.decode_fst = decode_fst;
        p.decodable_info = &decodable_info;
        p.trans_model = &trans_model;
        p.feature_info = &feature_info;
        p.decoder_opts = &decoder_opts;
        p.word_syms = word_syms;
        p.num_frames = &num_frames;
        p.tot_like = &tot_like;
        p.clat_write_mutex = &clat_write_mutex;
        p.write_lattice = write_lattice;
#if 1
        // enqueue work in thread pool
        futures.push_back(work_pool.enqueue(ProcessOneWave, p));
#else
        ProcessOneWave(p);
#endif
        num_task_submitted++;
        if (num_todo != -1 && num_task_submitted >= num_todo) break;
      }
    }

    for (int i = 0; i < futures.size(); i++) {
      futures[i].get();
    }

    // number of seconds elapsed since the creation of timer
    double total_time = timer.Elapsed();

    KALDI_LOG << "Decoded " << num_task_submitted << " utterances, " << num_err
              << " with errors.";
    KALDI_LOG << "Overall likelihood per frame was " << (tot_like / num_frames)
              << " per frame over " << num_frames << " frames.";
    KALDI_LOG << "Overall: "
              << " Aggregate Total Time: " << total_time
              << " Total Audio: " << total_audio * iterations
              << " RealTimeX: " << total_audio * iterations / total_time;

    clat_writer.Close();

    delete decode_fst;
    delete word_syms;  // will delete if non-NULL.
    return (num_task_submitted != 0 ? 0 : 1);
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}  // main()
