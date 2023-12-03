#!/bin/bash
#SBATCH --job-name=spark_job          # Job name
#SBATCH --nodes=4                     # Number of nodes to request
#SBATCH --ntasks-per-node=8           # Number of processes per node
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G                     # Memory per node
#SBATCH --time=24:00:00               # Maximum runtime in HH:MM:SS
#SBATCH --account=open                # Queue
#SBATCH --mail-user=ejs6233@psu.edu
#SBATCH --mail-type=ALL

# Load necessary modules (if required)
module load anaconda3
source activate ds410_f23
module use /gpfs/group/RISE/sw7/modules
module load spark/3.3.0
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=python3

TEC=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE * $SLURM_CPUS_PER_TASK))
EM=$(($SLURM_MEM_PER_NODE / $SLURM_NTASKS_PER_NODE / 1024))
echo "total-executor-cores=${TEC}"
echo "executor-memory=${EM}"

# Run PySpark
# Record the start time
start_time=$(date +%s)

spark-start
echo $MASTER | tee master.txt

spark-submit --total-executor-cores ${TEC} --executor-memory ${EM}G --driver-memory 5G scriptCrashlytics.py

#python scriptCrashlytics.py

# Record the end time
end_time=$(date +%s)

# Calculate and print the execution time
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time seconds"
