#!/bin/bash
#SBATCH --job-name=spark_job          # Job name
#SBATCH --nodes=4                     # Number of nodes to request
#SBATCH --ntasks-per-node=4           # Number of processes per node
#SBATCH --mem=16G                     # Memory per node
#SBATCH --time=3:00:00                # Maximum runtime in HH:MM:SS
#SBATCH --account=open 	              # Queue
#SBATCH --mail-user=ejs6233@psu.edu
#SBATCH --mail-type=ALL

# Load necessary modules (if required)
module load anaconda3
source activate ds410_f23
module use /gpfs/group/RISE/sw7/modules
module load spark/3.3.0
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=python3

# Run PySpark
# Record the start time
start_time=$(date +%s)
spark-submit --deploy-mode client --driver-memory 14G scriptCrashlytics.py

#python scriptCrashlytics.py

# Record the end time
end_time=$(date +%s)

# Calculate and print the execution time
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time seconds"
