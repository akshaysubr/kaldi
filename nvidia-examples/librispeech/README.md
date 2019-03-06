This example is an english model trained on the LibriSpeech dataset.  

To run this example you must first prepare the model:

%> ./prepare.sh


Once you have prepared the model you can run the model using our benchmark 
script:

%> ./benchmark_decoder.sh

This will run across two test datasets: test-clean and test-other.

Expected results on V100-SXM-16GB are the following:

test-clean:

Iteration: 1 ~Aggregate Total Time: 10.2769 Total Audio: 19229.6 RealTimeX: 1871.14
Iteration: 2 ~Aggregate Total Time: 15.9921 Total Audio: 38459.1 RealTimeX: 2404.89
Iteration: 3 ~Aggregate Total Time: 22.6928 Total Audio: 57688.7 RealTimeX: 2542.16
Iteration: 4 ~Aggregate Total Time: 29.2389 Total Audio: 76918.3 RealTimeX: 2630.69
Iteration: 5 ~Aggregate Total Time: 35.8846 Total Audio: 96147.9 RealTimeX: 2679.37
Iteration: 6 ~Aggregate Total Time: 42.4694 Total Audio: 115377 RealTimeX: 2716.72
Iteration: 7 ~Aggregate Total Time: 49.1109 Total Audio: 134607 RealTimeX: 2740.88
Iteration: 8 ~Aggregate Total Time: 55.671 Total Audio: 153837 RealTimeX: 2763.32
Iteration: 9 ~Aggregate Total Time: 61.7413 Total Audio: 173066 RealTimeX: 2803.08
Iteration: 10 ~Aggregate Total Time: 66.9026 Total Audio: 192296 RealTimeX: 2874.26
Overall:  Aggregate Total Time: 66.9027 Total Audio: 192296 RealTimeX: 2874.26
  %WER 14.00 [ 7329 / 52343, 860 ins, 725 del, 5744 sub ]
  %SER 74.00 [ 2175 / 2939 ]
  Scored 2939 sentences, 0 not present in hyp.
  Expected: 2939, Actual: 2939
  Decoding completed successfully.

test-other:

Iteration: 1 ~Aggregate Total Time: 10.2769 Total Audio: 19229.6 RealTimeX: 1871.14
Iteration: 2 ~Aggregate Total Time: 15.9921 Total Audio: 38459.1 RealTimeX: 2404.89
Iteration: 3 ~Aggregate Total Time: 22.6928 Total Audio: 57688.7 RealTimeX: 2542.16
Iteration: 4 ~Aggregate Total Time: 29.2389 Total Audio: 76918.3 RealTimeX: 2630.69
Iteration: 5 ~Aggregate Total Time: 35.8846 Total Audio: 96147.9 RealTimeX: 2679.37
Iteration: 6 ~Aggregate Total Time: 42.4694 Total Audio: 115377 RealTimeX: 2716.72
Iteration: 7 ~Aggregate Total Time: 49.1109 Total Audio: 134607 RealTimeX: 2740.88
Iteration: 8 ~Aggregate Total Time: 55.671 Total Audio: 153837 RealTimeX: 2763.32
Iteration: 9 ~Aggregate Total Time: 61.7413 Total Audio: 173066 RealTimeX: 2803.08
Iteration: 10 ~Aggregate Total Time: 66.9026 Total Audio: 192296 RealTimeX: 2874.26
Overall:  Aggregate Total Time: 66.9027 Total Audio: 192296 RealTimeX: 2874.26
  %WER 14.00 [ 7329 / 52343, 860 ins, 725 del, 5744 sub ]
  %SER 74.00 [ 2175 / 2939 ]
  Scored 2939 sentences, 0 not present in hyp.
  Expected: 2939, Actual: 2939
  Decoding completed successfully.

