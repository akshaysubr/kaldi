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
These examples are untested and are not guarenteeded to work.  

We have also provided a LibriSpeech example that has been tested and tuned
on the DGX1V platform.  

## LibriSpeech Example

An example has been provided and can be found here:
    nvidia-examples/librispeech/

To run the example you will first have to prepare the model

```
cd /workspace/nvidia-examples/librispeech/
./prepare.sh 
```

Once the model is prepared you can run a speech to text benchmark as follows:

```
cd /workspace/nvidia-examples/librispeech/
./run_benchmark.sh
```

To run on multiple-GPUs you must run a separate application on each GPU.  We have 
provided a script demonstrating this.

```
cd /workspace/nvidia-examples/librispeech/
./run_multigpu_benchmark.sh
```

## Customizing model parameters

Model parameters are specified in two files.  First global parameters are found 
in /workspace/nvidia-examples/default_parameters.inc.  Second model specific overrides
of the defaults can be found in /workspace/nvidia-examples/<model_director>/default_parameters.inc.
In addition, any parameter settings can be overridden by setting the parameter in the users
enviorment prior to launching run_benchmark.sh.

## Running with a custom model or dataset

To run a custom model or dataset a user will need to copy that model and or dataset into the container.
It is suggested to copy it into a new directory under /models/<model_name> and /datasets/<model_name>.
To use our benchmark script the data must be in a .wav format which matches the model.  

Once the dataset and/or model is in place a user should create a new directory /workspace/nvidia-examples/<model_name>.
They should then add a symbolic link to the run scripts:

```
ln -s ../run_benchmark.sh
ln -s ../run_multigpu_benchmark.sh
```

Finally they will need to create a file default_parameters.inc in this model and set any parameters necessary.
This file must specify MODEL_PATH, DATASET_PATH, and DATASETS.

Please see /workspace/nvidia-examples/librispeech/default_parameters.inc for an example. 

## Suggested Reading

The open-source project can be found here: https://github.com/kaldi-asr/kaldi

Documentation can be found here:  http://kaldi-asr.org/



