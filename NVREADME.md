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
docker pull nvcr.io/nvidia/kaldi:<xx.xx>-py3, where <xx.xx>-py3 varies with release but will resemble something like 19.08-py3
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
3. `run_benchmark.sh` - used to decode speech given a target model and dataset while providing performance and accuracy metrics.  Advanced features of this script are described below.

## Running the Provided LibriSpeech Example

To quickly get started with Kaldi and to realize GPU inferencing performance on the LibriSpeech corpus, NVIDIA has provided an example in `/workspace/nvidia-examples/librispeech`. Within this directory, you will notice:
1. `default_parameters.inc` - used to define local parameters (e.g. location of datasets and models)
2. `prepare_data.sh` - a script used to automatically download a corpus of recorded text, translate the format from flac to wav and adjust the sample rate to 16kHz, download a pre-trained LibriSpeech model, and link the `run_benchmark.sh` script to the local folder

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

## run_benchmark.sh Advanced Usage

This script is designed to show how to run Kaldi and benchmark decoding.  It is not designed to demonstrate how to use Kaldi in production.  For production use a user should expect to have to do some additional work to meet their needs.

The benchmark is highly flexible and is controlled via enviornment variables.  These can be set prior to running the script like this:

```
USE_GPU=false NUM_PROCESSES=8 ./run_benchmark.sh
```

For a full list of variables please look at run_benchmark.sh.  Here is a breif list of important variables:

1. USE_GPU:  If set to to false the CPU benchmark will be ran.
2. NUM_PROCESSES:  Number of decoding processes to run.  These will do the same work for benchmarking purposes.  If USE_GPU=true then each process runs on a seperate GPU.
3. CPU_THREADS:  Number of CPU threads to use in the GPU decoder.
4. GPU_FEATURE:  If set to true GPU feature extraction will be enabled leading to better multi-GPU scalability.
5. MAX_BATCH_SIZE:  Number of wavs to attempt to batch together into a single processing flow.
6. ITERATIONS:  Number of times to decode the input set for timing.
7. MODEL_PATH:  Path to the model
8. MODEL_NAME:  Name for the model
9. DATASET:  Path to the dataset
10. OUTPUT_PATH:  Base path for output
11. SEGMENT_SIZE:  Audio of this length will be split up into smaller segments around the size of SEGMENT_SIZE.

This benchmark can run single or multi-process, split audio into segments, and run CPU or GPU.  The behavior of this script is controlled via enviornment variables.  For example USE_GPU=false ./run_benchmark.sh would run on the CPU where USE_GPU=true ./run_benchmark.sh would run on the GPU. See the script for a complete list of varibales that control execution.

### Output

Output is written by default to $RESULT_PATH where RESULT_PATH=$OUTPUT_PATH/$RUN/$PROCESS, where RUN is the unique run number which starts at zero and increments by one each time the benchmark is ran with that OUTPUT_PATH, and PROCESS is the unqiue process id (0 to $NUM_PROCESSES).

The unique output directory contains a number of files releated to decoding including the dataset that was transcribed ($RESULT_PATH/dataset), the WER rate ($RESULT_PATH/wer), the RTF score ($RESULT_PATH/rtf), and the transcription ($RESULT_PATH/trans).

### How to Use Your Own Data with Kaldi
To stress a point made earlier in the README, if you're new to Kaldi and want to experiment with your own data, we highly recommend reading the Kaldi documentation, especially the [Kaldi for Dummies Tutorial](http://kaldi-asr.org/doc/kaldi_for_dummies.html). This section is not an exhaustive explanation of the ins-and-outs of Kaldi and is strictly focused on speech-to-text *inference*. As such, we do not refer to Kaldi standards like `utt2spk` that tells what recording belongs to which speaker or `text` that gives the ground-truth transcription for a recording. Both these files apply to training. Further, Kaldi (and NVIDIA's GPU acceleration work) shines on large corpuses of recorded speech. For this tutorial, we'll focus on a handful of recordings to easier demonstrate what's needed as one scales up for real-life applications and performance.

In general, each speech-to-text **inferencing** application will need the following folders and data:
1. `data` - a collection of .wav recordings of human speech
2. `datasets` - contains, for this container, a wav_conv.scp file that is structured as <file_name> <file_location> for each wav file to be transcribed
3. `models` - pre-trained model for a given speech corpus (LibriSpeech or ASpIRE, for example)

If you're recording your own audio, ensure that you sample at 16kHz over a Mono channel and export the audio as a wav file. [Audacity](https://www.audacityteam.org/download/) is a good tool to get started with your own recordings. You'll need to construct your own wav.scp file. 

An example is provided below:
```
1089-134686-0000 /workspace/data//LibriSpeech/test-clean/1089/134686/1089-134686-0000.wav
1089-134686-0001 /workspace/data//LibriSpeech/test-clean/1089/134686/1089-134686-0001.wav
1089-134686-0002 /workspace/data//LibriSpeech/test-clean/1089/134686/1089-134686-0002.wav
1089-134686-0003 /workspace/data//LibriSpeech/test-clean/1089/134686/1089-134686-0003.wav
1089-134686-0004 /workspace/data//LibriSpeech/test-clean/1089/134686/1089-134686-0004.wav
```

If the $DATASET directory does not contain a wav.scp file then one will be automatically created with all wav files in the folder or subfolders provided. 

To mount local folders to your Kaldi Docker container, run the following:
```
docker run --rm -it -v <path/to/local/datasets>:/workspace/datasets/<model_name>/<recordings_name> -v </path/to/local/data>:/workspace/data/<model_name>/<recordings_name> -v </path/to/local/models>:/workspace/models/<model_name> nvcr.io/nvidia/kaldi:<xx.xx>-py3
```
If you want to make use of the LibriSpeech model on your recordings stored under "transcribe_this", <model_name> would be `LibriSpeech` and <recordings_name> would be `transcribe_this`.

We're still going to leverage the `run_benchmark.sh` script contained in the `/workspace/nvidia-examples` folder. 

Create a new directory, `/workspace/nvidia-examples/<model_name>` and then add a symbolic link to the run scripts. We'll also need to copy a `default_parameters.inc` file from an existing project:
```
ln -s ../run_benchmark.sh
cp /workspace/nvidia-examples/librispeech/default_parameters.inc /workspace/nvidia-examples/<model_name>
```
Edit the `default_parameters.inc` folder in `/workspace/nvidia-examples` to ensure the correct MODEL_PATH and DATASET are set.

Once the dataset and/or model is in place, a user should create a new directory /workspace/nvidia-examples/<model_name>
and then add a symbolic link to the run scripts.

If you have truth data avaialable it should be placed in $DATASET/text.  If this file is not present then scoring of the result will not take place.

Run the benchmark as before:
```
cd /workspace/nvidia-examples/<model_name>/
./run_benchmark.sh
```

## Suggested Reading

The open-source project can be found here: [https://github.com/kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi)

Documentation can be found here: [http://kaldi-asr.org/](http://kaldi-asr.org/)



