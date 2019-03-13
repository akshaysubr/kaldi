KALDI
============

Kaldi is an open-source software framework for speech processing.  

## Contents of the Kaldi image

This container has Kaldi pre-built and ready to use in /opt/kaldi. In addition,
the source can be found in /opt/kaldi/src.

The open-source project can be found here: https://github.com/kaldi-asr/kaldi
Documentation can be found here:  http://kaldi-asr.org/

Kaldi is pre-built in this container, but it can be rebuilt like this:

```
%> make -C -j /opt/kaldi/src/
```

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



