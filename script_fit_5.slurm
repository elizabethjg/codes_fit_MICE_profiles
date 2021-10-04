#!/bin/bash
#SBATCH --job-name=fit5
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

srun python -u fit_MICE_profiles.py -file '5_1'
srun python -u fit_MICE_profiles.py -file '5_2'
srun python -u fit_MICE_profiles.py -file '5_3'
srun python -u fit_MICE_profiles.py -file '5_4'
srun python -u fit_MICE_profiles.py -file '5_5'
srun python -u fit_MICE_profiles.py -file '5_6'
srun python -u fit_MICE_profiles.py -file '5_7'
srun python -u fit_MICE_profiles.py -file '5_8'
srun python -u fit_MICE_profiles.py -file '5_9'
srun python -u fit_MICE_profiles.py -file '5_10'
srun python -u fit_MICE_profiles.py -file '7_1'
srun python -u fit_MICE_profiles.py -file '7_2'
srun python -u fit_MICE_profiles.py -file '7_3'

