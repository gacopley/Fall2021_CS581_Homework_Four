#!/bin/bash
module load openmpi/4.0.5-gnu-pmi2
srun --mpi=pmi2 ./life -n 5000 -max 5000 -fo /scratch/$USER/
srun --mpi=pmi2 ./life -n 5000 -max 5000 -fo /scratch/$USER/
srun --mpi=pmi2 ./life -n 5000 -max 5000 -fo /scratch/$USER/