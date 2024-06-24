#!/bin/sh
#SBATCH --job-name=multicore_job
#SBATCH --nodes=1
#SBATCH --partition v100_pree
#SBATCH --gres=gpu:3
#SBATCH --time=80:99:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=170G

module load python/3.7-Anaconda-datalad

eval "$(conda shell.bash hook)"

conda activate /dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/alos/ad_ml 

python /dartfs-hpc/rc/home/k/f006fpk/scratch/antibody/working_dir/bert_DNA_model.py
