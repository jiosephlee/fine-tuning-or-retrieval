#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --partition=dgx-b200
#SBATCH --output=test.%j.out

echo "hello from $(hostname)"
