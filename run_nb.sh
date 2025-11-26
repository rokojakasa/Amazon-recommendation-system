#!/bin/bash
#BSUB -J minhash_job                     # Job name
#BSUB -o minhash_job.out                 # Output log file
#BSUB -e minhash_job.err                 # Error log file
#BSUB -q hpc                           # Queue (no GPU needed)
#BSUB -n 8                                # Number of CPU cores
#BSUB -R "rusage[mem=4G]"                # Memory per core
#BSUB -R "span[hosts=1]"                 # All cores on same node
#BSUB -W 12:00                            # Max runtime 12 hours

module load python3
source ~/minhash_env/bin/activate

cd /zhome/4f/1/223566/home/computational_tools/Amazon-recommendation-system
export PYTHONPATH=$(pwd):$PYTHONPATH
python src/recommender_lsh.py

