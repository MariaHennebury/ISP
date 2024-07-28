#!/bin/bash
#SBATCH --job-name=MHennebury_ISP
#SBATCH --account=an-tr043
#SBATCH --time=1:00:00   # Total runtime, adjust as needed
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=MHennebury-%j.out  # Redirects stdout and stderr to the same file

# Load the required modules
module load python/3.11.5

# Set up the virtual environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Upgrade pip and install required packages
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

# List installed packages to verify installation
pip list

# Run your Python script
python ISP.py

# Display the hostname of the compute node
srun hostname