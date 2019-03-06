KALDI
============

Kaldi is an open source software framework for speech processing.  

## Contents of the Kaldi image

This container has Kaldi pre-built and ready to use in /opt/kaldi. In addition 
the source can be found in /opt/kaldi/src.  This source currently has changes 
that are not included in the kaldi mainline but will be soon.  

The mainline can be found here: https://github.com/kaldi-asr/kaldi

Kaldi documentation can be found here:  http://kaldi-asr.org/

Kaldi is pre-built but can be rebuilt like this:

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
./benchmark_decoder.sh
```




