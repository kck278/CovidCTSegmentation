#!/bin/bash
#SBATCH --ntasks=20
#SBATCH --time=1:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=segnet
# python trainer.py -m "SegNet" -c 4 -b 4 -l 1e-4
# python trainer.py -m "SegNet" -c 4 -b 8 -l 1e-4
# python trainer.py -m "SegNet" -c 4 -b 12 -l 1e-4
# python trainer.py -m "SegNet" -c 4 -b 4 -l 1e-3
# python trainer.py -m "SegNet" -c 4 -b 8 -l 1e-3
# python trainer.py -m "SegNet" -c 4 -b 12 -l 1e-3
# python trainer.py -m "SegNet" -c 4 -b 4 -l 3e-3
# python trainer.py -m "SegNet" -c 4 -b 8 -l 3e-3
# python trainer.py -m "SegNet" -c 4 -b 12 -l 3e-3


python trainer.py -m "SegNetOriginal" -c 2 -b 4 -l 1e-4
python trainer.py -m "SegNetOriginal" -c 2 -b 8 -l 1e-4
python trainer.py -m "SegNetOriginal" -c 2 -b 12 -l 1e-4
python trainer.py -m "SegNetOriginal" -c 2 -b 4 -l 1e-3
python trainer.py -m "SegNetOriginal" -c 2 -b 8 -l 1e-3
python trainer.py -m "SegNetOriginal" -c 2 -b 12 -l 1e-3
python trainer.py -m "SegNetOriginal" -c 2 -b 4 -l 3e-3
python trainer.py -m "SegNetOriginal" -c 2 -b 8 -l 3e-3
python trainer.py -m "SegNetOriginal" -c 2 -b 12 -l 3e-3

python trainer.py -m "SegNetOriginal" -c 4 -b 4 -l 1e-4
python trainer.py -m "SegNetOriginal" -c 4 -b 8 -l 1e-4
python trainer.py -m "SegNetOriginal" -c 4 -b 12 -l 1e-4
python trainer.py -m "SegNetOriginal" -c 4 -b 4 -l 1e-3
python trainer.py -m "SegNetOriginal" -c 4 -b 8 -l 1e-3
python trainer.py -m "SegNetOriginal" -c 4 -b 12 -l 1e-3
python trainer.py -m "SegNetOriginal" -c 4 -b 4 -l 3e-3
python trainer.py -m "SegNetOriginal" -c 4 -b 8 -l 3e-3
python trainer.py -m "SegNetOriginal" -c 4 -b 12 -l 3e-3
#