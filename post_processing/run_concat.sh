#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH -o /mnt/home/njoseph/gily_exp/post_processing/sync_logs/concat_log_%j.log
pwd; hostname; date;

##################################################
# Usage:
# sbatch /mnt/home/njoseph/gily_exp/post_processing/run_concat.sh
##################################################

echo "Setting up environment"
source /mnt/home/njoseph/miniforge3/bin/activate;
conda activate common;
echo "Running concatenation"
python /mnt/home/njoseph/gily_exp/post_processing/concatenate_data_cam_mic_sync_gily_automated_flatiron.py "$@"
date;