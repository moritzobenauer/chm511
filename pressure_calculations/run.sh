#!/bin/bash
#MLO 2024 @ Princeton University

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
