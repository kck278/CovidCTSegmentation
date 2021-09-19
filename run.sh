#!/bin/bash
#SBATCH --ntasks=20
#SBATCH --time=12:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:8
#SBATCH --job-name=unet

python trainer.py -m "UNet" -c 2 -b 2 -l 1e-4 -e 160
python trainer.py -m "UNet" -c 2 -b 8 -l 1e-4 -e 160
python trainer.py -m "UNet" -c 2 -b 12 -l 1e-4 -e 160
python trainer.py -m "UNet" -c 2 -b 2 -l 5e-4 -e 160
python trainer.py -m "UNet" -c 2 -b 8 -l 5e-4 -e 160
python trainer.py -m "UNet" -c 2 -b 12 -l 5e-4 -e 160
python trainer.py -m "UNet" -c 2 -b 2 -l 1e-3 -e 160
python trainer.py -m "UNet" -c 2 -b 8 -l 1e-3 -e 160
python trainer.py -m "UNet" -c 2 -b 12 -l 1e-3 -e 160

python trainer.py -m "UNet" -c 4 -b 2 -l 1e-4 -e 160
python trainer.py -m "UNet" -c 4 -b 8 -l 1e-4 -e 160
python trainer.py -m "UNet" -c 4 -b 12 -l 1e-4 -e 160
python trainer.py -m "UNet" -c 4 -b 2 -l 5e-4 -e 160
python trainer.py -m "UNet" -c 4 -b 8 -l 5e-4 -e 160
python trainer.py -m "UNet" -c 4 -b 12 -l 5e-4 -e 160
python trainer.py -m "UNet" -c 4 -b 2 -l 1e-3 -e 160
python trainer.py -m "UNet" -c 4 -b 8 -l 1e-3 -e 160
python trainer.py -m "UNet" -c 4 -b 12 -l 1e-3 -e 160

python trainer.py -m "UNet" -c 2 -b 2 -l 1e-4 -e 160 -ext True
python trainer.py -m "UNet" -c 2 -b 8 -l 1e-4 -e 160 -ext True
python trainer.py -m "UNet" -c 2 -b 12 -l 1e-4 -e 160 -ext True
python trainer.py -m "UNet" -c 2 -b 2 -l 5e-4 -e 160 -ext True
python trainer.py -m "UNet" -c 2 -b 8 -l 5e-4 -e 160 -ext True
python trainer.py -m "UNet" -c 2 -b 12 -l 5e-4 -e 160 -ext True
python trainer.py -m "UNet" -c 2 -b 2 -l 1e-3 -e 160 -ext True
python trainer.py -m "UNet" -c 2 -b 8 -l 1e-3 -e 160 -ext True
python trainer.py -m "UNet" -c 2 -b 12 -l 1e-3 -e 160 -ext True

python trainer.py -m "UNet" -c 4 -b 2 -l 1e-4 -e 160 -ext True
python trainer.py -m "UNet" -c 4 -b 8 -l 1e-4 -e 160 -ext True
python trainer.py -m "UNet" -c 4 -b 12 -l 1e-4 -e 160 -ext True
python trainer.py -m "UNet" -c 4 -b 2 -l 5e-4 -e 160 -ext True
python trainer.py -m "UNet" -c 4 -b 8 -l 5e-4 -e 160 -ext True
python trainer.py -m "UNet" -c 4 -b 12 -l 5e-4 -e 160 -ext True
python trainer.py -m "UNet" -c 4 -b 2 -l 1e-3 -e 160 -ext True
python trainer.py -m "UNet" -c 4 -b 8 -l 1e-3 -e 160 -ext True
python trainer.py -m "UNet" -c 4 -b 12 -l 1e-3 -e 160 -ext True