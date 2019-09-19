import subprocess
import re
import random
import os
import sys

if sys.version_info[0] < 3:
    raise Exception("search.py requries Python 3")

debug = False

wer_regex = re.compile('WER ([0-9.]*) ')
rtf_regex = re.compile('RTF: ([0-9.]*) ')

best_rtf = 0.0
max_wer = 1000
env = {}
env['BEAM']="15"
env['LATTICE_BEAM']="2.5"
env['GPU_THREADS']="3"
env['COPY_THREADS']="2"
env['MAX_ACTIVE']="6000"
env['AUX_Q_CAPACITY']="500000"
env['MAIN_Q_CAPACITY']="40000"
env['MAX_BATCH_SIZE']="50"
env['BATCH_DRAIN_SIZE']="5"
try:
   infile = open("best_params.inc",'r')
   for line in infile:
       foo=line.split("=")
       foo[0] = foo[0].rstrip().lstrip()
       foo[1] = foo[1].rstrip().lstrip()
       print("Setting " + foo[0] +" to " + foo[1])
       env[foo[0]] = foo[1]
except:
   print("Couldn't find or open best_params.inc")
if 'WER' in env:
    max_wer = float(env['WER'])
    env.pop('WER')
if 'RTF' in env:
    best_rtf = float(env['RTF'])
    env.pop('RTF')
adjustable_vars = env.keys()
int_vars = ['MAX_ACTIVE','MAIN_Q_CAPACITY','AUX_Q_CAPACITY','MAX_BATCH_SIZE',
            'BATCH_DRAIN_SIZE','COPY_THREADS','GPU_THREADS','BATCH_SIZE']

full_env = {}
for k in os.environ.keys():
    full_env[k] = os.environ[k]

adjustment = 0.0
while(1):

   #Make a small change to env
   key = random.choice(list(env.keys()))
   last_val = env[key]
   if key in int_vars:
      if adjustment > 0:
         env[key] = str(int(env[key]) + int(env[key])//10 + 1)
      else:
         env[key] = str(int(env[key]) - int(env[key])//10 - 1)
   else:
      env[key]=str(float(env[key])*(1+adjustment))
   print("env["+key+"] = " + str(env[key]))
   #try:
   for k in env.keys():
       full_env[k] = env[k]
   try:
       resp = subprocess.run(["./run_benchmark.sh"], stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE, env=full_env)
       wer = float(wer_regex.search(str(resp.stdout)).group(1))
       rtf = float(rtf_regex.search(str(resp.stdout)).group(1))
       print("WER: " + str(wer) + ".  RTF: " + str(rtf))
       if max_wer==1000:
           max_wer = wer
   except:
       print("Run FAILED *****************************")
       wer = max_wer+1
   if (wer <= max_wer and rtf > best_rtf):
       best_rtf = rtf
       print ("****  Found new best RTF: " + str(rtf) + " ********")
       print (env)
       output_file = open("best_params.inc",'w')
       for k in env:
           output_file.write(k + "=" + str(env[k]) + "\n")
       output_file.write("WER=" + str(max_wer)+"\n")
       output_file.write("RTF=" + str(rtf)+"\n")
       output_file.close()
   else:
       env[key]=last_val
   if debug:
       print(env)
       print("Best RTF = " + str(best_rtf))
       print("Max WER = " + str(max_wer))
   adjustment = (random.random()-0.5)*0.2
   



#resp = subprocess.run(["./run_benchmark.sh"], capture_output=True)
#print(resp.stdout)


