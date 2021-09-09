#!/bin/bash
#SBATCH --ntasks=20
#SBATCH --time=2:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=unet
python trainer.py