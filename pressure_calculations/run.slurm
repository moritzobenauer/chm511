#!/bin/bash
#MLO 2024 @ Princeton University

#SBATCH --job-name=lammpst1      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=03:59:59          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=mo9089@princeton.edu
#SBATCH --constraint=cascade


RHO=0.4984
STEPS_EQ=20000
STEPS_PROD=2000

module purge
module load intel/2024.0
module load intel-mpi/intel/2021.7.0
module load intel-mkl/2024.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

srun $HOME/.local/bin/lmp_intel -sf intel -in yukawa_longbox.lmp -var rho $RHO  -var steps $STEPS_EQ
srun $HOME/.local/bin/lmp_intel -sf intel -in calcpressure.lmp -var rho $RHO -var steps $STEPS_PROD
