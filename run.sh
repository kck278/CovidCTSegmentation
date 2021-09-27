#!/bin/bash
#SBATCH --ntasks=20
#SBATCH --time=10:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:4
#SBATCH --job-name=segnet_modified_dataset

python trainer.py -m "SegNet" -c 2 -b 12 -l 1e-4 -e 160 -r 512 # 3
python trainer.py -m "SegNet" -c 4 -b 4 -l 3e-3 -e 160 -r 512 # 7

python trainer.py -m "SegNet" -c 2 -b 12 -l 1e-4 -e 160 -ext True # 3
python trainer.py -m "SegNet" -c 4 -b 4 -l 3e-3 -e 160 -ext True # 7

python trainer.py -m "SegNet" -c 2 -b 12 -l 1e-4 -e 160 -ext True -r 512 # 3
python trainer.py -m "SegNet" -c 4 -b 4 -l 3e-3 -e 160 -ext True -r 512 # 7
