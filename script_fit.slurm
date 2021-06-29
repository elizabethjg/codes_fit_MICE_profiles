#!/bin/bash
#SBATCH --job-name=fitpro
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=55
#SBATCH --time=3:00:00
## An email is sent when the job is canceled or finished
## Change "root" for your e-mail address
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ejgonzalez@unc.edu.ar


. /etc/profile
source $HOME/.bashrc

cd /mnt/projects/lensing/HALO_SHAPE/MICEv2.0/codes_fit_MICE_profiles/

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

conda activate py3env

srun python -u fit_MICE_profiles.py
