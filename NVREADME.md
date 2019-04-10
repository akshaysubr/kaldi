KALDI
============

[Kaldi](http://kaldi-asr.org/) is an open-source software framework for speech processing that originated in 2009 at [Johns Hopkins University](https://www.jhu.edu/) with the intent to develop techniques to reduce both the cost and time required to build speech recognition systems. Kaldi has since grown to become the de-facto speech recognition toolkit in the community, helping enable speech services used by millions of people each day.

While Kaldi adopted GPU acceleration for speech training workloads early in the project's development, in 2017, NVIDIA - in conjunction with Johns Hopkins - focused on better utilization of GPUs for speech-to-text inference acceleration. Through the introduction of a GPU-based Viterbi decoder, NVIDIA demonstrated up to 3524x real time speech-to-text transcription performance on a Tesla V100. This exceeds a 2-Socket Intel Xeon Platinum 8168 CPU implementation by a factor of 9.2.

You can learn more about NVIDIA's work in GPU-accelerating speech-to-text inference along with our performance results by reading our blog post [here](https://devblogs.nvidia.com/nvidia-accelerates-speech-text-transcription-3500x-kaldi/).

## Overview

This README seeks to both guide you through NVIDIA's speech-to-text performance benchmarks using the LibriSpeech corpus on both clean and noisy speech recordings and also show you how to generate transcripts from your own recorded speech. In the case of using your own recordings, performance is not guaranteed and varies with respect to human accent, distance from the microphone, background noise, language, and the accuracy of the trained model used to perform inference. Nevertheless, we hope that this guide is helpful for the full range of Kaldi developers.

The README is structured as follows:
1. How to Pull and Run the Kaldi Container
2. Overview of the Kaldi Docker Image
3. Running the Provided LibriSpeech Example
4. Generating Transcripts from Decoded Speech
5. How to Use Your Own Data with Kaldi

## Pulling and Running the Kaldi Container

The [Kaldi container](https://ngc.nvidia.com/catalog/containers/nvidia:kaldi) is hosted on [NVIDIA's GPU Cloud](https://ngc.nvidia.com). If you do not have an account, you'll need to create one to pull the Kaldi container.

Before pulling the Kaldi container, ensure that you have successfully installed a recent NVIDIA GPU Driver and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). NVIDIA Docker is a wrapper on Docker-CE that exposes the graphics driver to the container.

### Pull the container:
```
docker pull nvcr.io/nvidia/kaldi:<xx.xx>-py3, where <xx.xx>-py3 varies with release but will resemble something like 19.03-py3
```

### Run the container:
```
nvidia-docker run -it --rm nvcr.io/nvidia/kaldi:<xx.xx>-py3
```

If you've installed nvidia-docker2, you can simply run:
```
docker run --runtime=nvidia -it --rm nvcr.io/nvidia/kaldi:<xx.xx>-py3
```

Or, if you have the default Docker runtime set to NVIDIA:
```
docker run -it --rm nvcr.io/nvidia/kaldi:<xx.xx>-py3
```

## Overview of the Kaldi Docker Image

For those users familiar with the Kaldi toolkit, the full software suite comes pre-built within this container and is located in `/opt/kaldi`. In addition, the Kaldi source code can be found in `/opt/kaldi/src`.

While not necessary, Kaldi can be rebuilt within the container via:

```
%> make -j -C /opt/kaldi/src/
```

Standard Kaldi examples provided by Johns Hopkins can be found in `/workspace/examples`. These examples, however, are untested and are not guaranteed to work. For new Kaldi users, we recommend reading the [Kaldi for Dummies Tutorial](http://kaldi-asr.org/doc/kaldi_for_dummies.html). 

### NVIDIA Provided Examples

For examples that have been vetted and verified by NVIDIA, navigate to `/workspace/nvidia-examples`. This folder contains the following:
1. `default_parameters.inc` - used to define global default Kaldi and GPU specific parameters
2. `librispeech` - folder, used as the 'launchpad' for reproducing our performance benchmarks. More information is given in the "Running the Provided LibriSpeech Example" section below
3. `run_benchmark.sh` - used to decode speech given a target model and dataset while providing performance and accuracy metrics on a single GPU
4. `run_multigpu_benchmark.sh` - runs the above benchmark on multiple GPUs

## Running the Provided LibriSpeech Example

To quickly get started with Kaldi and to realize GPU inferencing performance on the LibriSpeech corpus, NVIDIA has provided an example in `/workspace/nvidia-examples/librispeech`. Within this directory, you will notice:
1. `default_parameters.inc` - used to define local parameters (e.g. location of datasets and models)
2. `prepare_data.sh` - a script used to automatically download a corpus of recorded text, translate the format from flac to wav and adjust the sample rate to 16kHz, download a pre-trained LibriSpeech model, and link the `run_benchmark.sh` and `run_multigpu_benchmark.sh` scripts to the local folder

It should be noted that our LibriSpeech example *should work out of the box* and no changes to default parameters or datapaths are necessary. Editing these files and understanding the Kaldi tools will be discussed in more depth with "How to Use Your Own Data with Kaldi".

To run the example, we first need to download example recordings, pre-process them, and download the trained model:
```
cd /workspace/nvidia-examples/librispeech/
./prepare.sh 
```

Once the `prepare.sh` script is complete, you can run the LibriSpeech speech to text benchmark as follows:
```
cd /workspace/nvidia-examples/librispeech/
./run_benchmark.sh
```

If you have multiple GPUs in your test system, you can invoke:
```
cd /workspace/nvidia-examples/librispeech/
./run_multigpu_benchmark.sh
```
Please note that we've supressed all intermediate text output with the multi-GPU benchmark. Running this command will take some time before it produces its brief summary.

GPU isolation within the Docker container can be manipulated with the `NVIDIA_VISIBLE_DEVICES` environment variable. For example:
```
export NVIDIA_VISIBLE_DEVICES=0,2
```
Makes only GPUs 0 and 2 visible to applications.

## Generating Transcripts from Decoded Speech

While performance benchmarks are presented on the terminal after the completion of the benchmark scripts, you can view the transcript output by navigating to `/tmp/ls-results.0`. There are 3 important files to consider in this directory:
1. `words.txt` - a dictionary of English language words and a corresponding index. For example, the word 'jump' is 92598 while 'zebra' is 199291
2. `trans.batched-wav-nnet3-cuda.test_clean.gz` - zipped decoded transcript that contains numerical values for decoded speech, in this case, for the test_clean dataset
3. `trans.batched-wav-nnet3-cuda.test_other.gz` - same as above but for the test_other (noisy) dataset

Generate transcripts by:
```
gunzip trans.batched-wav-nnet3-cuda.test_clean.gz
/opt/kaldi/egs/librispeech/s5/utils/int2sym.pl -f 2- words.txt trans.batched-wav-nnet3-cuda.test_clean >> transcript_test_clean
```
The `int2sym.pl` script maps the `words.txt` dictionary to the decoded numerical values found from the output of the decoded lattice. The 2- flag is file specific and instructs the skipping of the first item in each row (filename).

You can listen to the recorded speech examples by navigating to the downloaded data files, located in `/workspace/data/LibriSpeech/test-clean` or `/workspace/data/LibriSpeech/test-other`. Matching recordings to transcripts can be done by opening the `int2sym` output file and noting the first item in each row. For example:
```
1089-134686-0000 he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flower fattened sauce
1089-134686-0001 stuff it into you his belly counselled him
1089-134686-0002 after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels
1089-134686-0003 hello bertie any good in your mind
1089-134686-0004 number den fresh nellie is waiting on you good night husband
1089-134686-0005 the music came nearer and he recalled the words the words of shelley's fragment upon the moon wandering companionless pale for weariness
```
Where 1089-134686-0000 maps to `/workspace/data/LibriSpeech/test-clean/1089/134686/0000.wav`

## How to Use Your Own Data with Kaldi
To stress a point made earlier in the README, if you're new to Kaldi and want to experiment with your own data, we highly recommend reading the Kaldi documentation, especially the [Kaldi for Dummies Tutorial](http://kaldi-asr.org/doc/kaldi_for_dummies.html). This section is not an exhaustive explanation of the ins-and-outs of Kaldi and is strictly focused on speech-to-text *inference*. As such, we do not refer to Kaldi standards like `utt2spk` that tells what recording belongs to which speaker or `text` that gives the ground-truth transcription for a recording. Both these files apply to training. Further, Kaldi (and NVIDIA's GPU acceleration work) shines on large corpuses of recorded speech. For this tutorial, we'll focus on a handful of recordings to easier demonstrate what's needed as one scales up for real-life applications and performance.

In general, each speech-to-text **inferencing** application will need the following folders and data:
1. `data` - a collection of .wav recordings of human speech
2. `datasets` - contains, for this container, a wav_conv.scp file that is structured as <file_name> <file_location> for each wav file to be transcribed
3. `models` - pre-trained model for a given speech corpus (LibriSpeech or ASpIRE, for example)

If you're recording your own audio, ensure that you sample at 16kHz over a Mono channel and export the audio as a wav file. [Audacity](https://www.audacityteam.org/download/) is a good tool to get started with your own recordings. You'll need to construct your own wav_conv.scp file. An example is provided below:
```
1089-134686-0000 /workspace/data//LibriSpeech/test-clean/1089/134686/1089-134686-0000.wav
1089-134686-0001 /workspace/data//LibriSpeech/test-clean/1089/134686/1089-134686-0001.wav
1089-134686-0002 /workspace/data//LibriSpeech/test-clean/1089/134686/1089-134686-0002.wav
1089-134686-0003 /workspace/data//LibriSpeech/test-clean/1089/134686/1089-134686-0003.wav
1089-134686-0004 /workspace/data//LibriSpeech/test-clean/1089/134686/1089-134686-0004.wav
```

To mount local folders to your Kaldi Docker container, run the following:
```
docker run --rm -it -v <path/to/local/datasets>:/workspace/datasets/<model_name>/<recordings_name> -v </path/to/local/data>:/workspace/data/<model_name>/<recordings_name> -v </path/to/local/models>:/workspace/models/<model_name> nvcr.io/nvidia/kaldi:<xx.xx>-py3
```
If you want to make use of the LibriSpeech model on your recordings stored under "transcribe_this", <model_name> would be `LibriSpeech` and <recordings_name> would be `transcribe_this`.

We're still going to leverage the `run_benchmark.sh` and `run_multigpu_benchmark.sh` scripts contained in the `/workspace/nvidia-examples` folder. 

Create a new directory, `/workspace/nvidia-examples/<model_name>` and then add a symbolic link to the run scripts. We'll also need to copy a `default_parameters.inc` file from an existing project:
```
ln -s ../run_benchmark.sh
ln -s ../run_multigpu_benchmark.sh
cp /workspace/nvidia-examples/librispeech/default_parameters.inc /workspace/nvidia-examples/<model_name>
```
Edit the `default_parameters.inc` folder in `/workspace/nvidia-examples` to ensure the correct MODEL_PATH, DATASET_PATH, and DATASETS are set. While MODEL_PATH and DATASET_PATH are self explanatory, one will replace "test_clean" and "test_other" in DATASETS with the <recordings_name> chosen above.

Once the dataset and/or model is in place, a user should create a new directory /workspace/nvidia-examples/<model_name>
and then add a symbolic link to the run scripts:

If you don't have truth data available for your dataset, we suggest editing the `run_benchmark.sh` script and commenting out the sections calculating accuracy metrics (starting around line 85: calculate wer and ending before the benchmark summary). Run the benchmark as before:
```
cd /workspace/nvidia-examples/<model_name>/
./run_benchmark.sh
```
Results, as with the standard LibriSpeech example, will be written to `/tmp/ls-results.0`, and you can use the same `int2sym.pl` command as before to generate human-readable transcripts.

## Suggested Reading

The open-source project can be found here: [https://github.com/kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi)

Documentation can be found here: [http://kaldi-asr.org/](http://kaldi-asr.org/)



