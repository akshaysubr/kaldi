KALDI
============

Kaldi is an open-source software framework for speech processing.  

## Contents of the Kaldi image

This container has Kaldi pre-built and ready to use in /opt/kaldi. In addition,
the source can be found in /opt/kaldi/src.

Kaldi is pre-built in this container, but it can be rebuilt like this:

```
%> make -j -C /opt/kaldi/src/
```

Examples provided by Johns Hopkins can be found in /workspace/examples. 
These examples are untested and are not guaranteed to work.  

We have also provided a LibriSpeech example that has been tested and tuned
on the DGX-1V platform.  

## LibriSpeech Example

A LibriSpeech example has been provided and can be found here:
    nvidia-examples/librispeech/

To run the example, you will first have to prepare the model. This script downloads and pre-processes the LibriSpeech model and example recordings.

```
cd /workspace/nvidia-examples/librispeech/
./prepare.sh 
```

Once the model is prepared you can run a speech to text benchmark as follows:

```
cd /workspace/nvidia-examples/librispeech/
./run_benchmark.sh
```

Multi-GPU support consists of running a separate application on each GPU. We have 
provided a script demonstrating this.

```
cd /workspace/nvidia-examples/librispeech/
./run_multigpu_benchmark.sh
```

## Customizing model parameters

Model parameters are specified in two files.
1.  /workspace/nvidia-examples/default_parameters.inc
2.  /workspace/nvidia-examples/<model_directory>/default_parameters.inc

Global parameters are found in (1) while model specific parameters 
that override default ones can be found in (2).
Additionally, any parameter setting can be overridden by setting the parameters in the users'
enviorment prior to launching run_benchmark.sh.

## Running with a custom model or dataset

To run a custom model or dataset, a user will need to copy that model and/or dataset into the container.
It is suggested to copy it into a new directory under /models/<model_name> and /datasets/<model_name>.
To use our benchmark script, the data must be in a .wav format which matches the model. For example:

```
nvidia-docker run --rm -it -v <local path to models>:/models/<model name> -v <local path to dataset>:/datasets/<model_name> kaldi:19.03-py3
```

Once the dataset and/or model is in place, a user should create a new directory /workspace/nvidia-examples/<model_name>
and then add a symbolic link to the run scripts:

```
ln -s ../run_benchmark.sh
ln -s ../run_multigpu_benchmark.sh
```

Finally, the user will need to create a file default_parameters.inc in this model and set any parameters necessary.
This file must specify MODEL_PATH, DATASET_PATH, and DATASETS.

Please see /workspace/nvidia-examples/librispeech/default_parameters.inc for an example. 

## Suggested Reading

The open-source project can be found here: https://github.com/kaldi-asr/kaldi

Documentation can be found here:  http://kaldi-asr.org/



