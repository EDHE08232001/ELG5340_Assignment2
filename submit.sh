#!/bin/bash
#SBATCH --account=small
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/uottawa.o.univ/ehe058/CSI5340_Assignment2-main/out/job-%j.out
#SBATCH --error=/home/uottawa.o.univ/ehe058/CSI5340_Assignment2-main/out/job-%j.err

cd /home/uottawa.o.univ/ehe058/CSI5340_Assignment2-main
source /home/uottawa.o.univ/ehe058/CSI5340_Assignment2-main/myenv/local/bin/activate
python main.py
