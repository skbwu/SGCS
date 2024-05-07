#!/bin/bash
#SBATCH -J CNN # A single job name for the array
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # All cores on one machine
#SBATCH -p sapphire,shared # Partition
#SBATCH --mem 16000 # Memory request
#SBATCH -t 0-12:00 # (D-HH:MM)
#SBATCH -o /n/holyscratch01/kou_lab/swu/288/outputs/%j.out # Standard output
#SBATCH -e /n/holyscratch01/kou_lab/swu/288/errors/%j.err # Standard error
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=skylerwu@college.harvard.edu
#SBATCH --account=meng_lab

module load cmake/3.25.2-fasrc01
module load gcc/12.2.0-fasrc01

# ONLY RUN ONE SETTING BECAUSE CNN IS MORE EXPENSIVE!
conda run -n afterburner python3 cifar10_cnn_main.py $1 $2