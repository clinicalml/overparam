import numpy as np
import subprocess
import sys
import os

''' Runs $runs instances of train_em_hybrid.py, with output path given
    by $output_path/R$i, where $i is the run number (between 1 and $runs). '''


samples_path = sys.argv[1]
output_path = sys.argv[2]
dim_latent = int(sys.argv[3])
runs = int(sys.argv[4])

for i in range(1, runs + 1):
    os.system('mkdir ' + output_path + '/R' + str(i))

# cpu_limit = 30

processes = []
for i in range(1, runs + 1):
    p_now = subprocess.Popen(['python', 'train_em_hybrid.py', samples_path, output_path + '/R' + str(i), str(dim_latent), str(np.random.random_integers(0, 1000000))])
    processes.append(p_now)

    # if len(processes) == cpu_limit:
    #     for p in processes:
    #         p.wait()
    #     processes = []

for p in processes:
    p.wait()
