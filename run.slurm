#!/bin/bash
#SBATCH --partition=large
#SBATCH --job-name="AI_Ass1"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=44
#SBATCH --mem=64gb 
#SBATCH --time=7-00:00:00
#SBATCH --account=paucaGrp
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=guptd23@wfu.edu
#SBATCH --output="/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Assignement1/main_code/SLURM/%j.o"
#SBATCH --error="/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Assignement1/main_code/SLURM/%j.e"
# Note: SLURM has no batch input for cputime, excluding.
# Note: NCPUS directive is redundant from expected nodes#:ppn# input, excluding.

# Load software for computation


# Change directories to location of test_script.py
cd /deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Assignement1/Alzhemier_Code

# Execute test_script.py
python3 main.py


