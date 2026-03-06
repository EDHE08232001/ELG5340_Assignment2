#!/bin/bash
#SBATCH --account=small            # which resource account to bill
#SBATCH --partition=small          # which queue to join
#SBATCH --gres=gpu:1               # request 1 GPU
#SBATCH --time=24:00:00            # max runtime (HH:MM:SS)
#SBATCH --output=out/job-%j.out    # stdout goes here (%j = job ID)
#SBATCH --error=out/job-%j.err     # stderr goes here

cd $HOME/hw2                       # go to your project folder
source $HOME/myenv/bin/activate    # activate your Python environment
python main.py                     # run your code
