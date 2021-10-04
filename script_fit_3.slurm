#!/bin/bash
#SBATCH --job-name=fit3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=1-10:00:00
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

srun python -u fit_MICE_profiles.py -file '3_1'
srun python -u fit_MICE_profiles.py -file '3_2'
srun python -u fit_MICE_profiles.py -file '3_3'
srun python -u fit_MICE_profiles.py -file '3_4'
srun python -u fit_MICE_profiles.py -file '3_5'
srun python -u fit_MICE_profiles.py -file '3_6'
srun python -u fit_MICE_profiles.py -file '3_7'
srun python -u fit_MICE_profiles.py -file '3_8'
srun python -u fit_MICE_profiles.py -file '3_9'
srun python -u fit_MICE_profiles.py -file '3_10'
srun python -u fit_MICE_profiles.py -file '9_1'
srun python -u fit_MICE_profiles.py -file '9_2'
srun python -u fit_MICE_profiles.py -file '9_3'
srun python -u fit_MICE_profiles.py -file '9_4'
