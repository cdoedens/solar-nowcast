import subprocess
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import sys

script_to_run = sys.argv[1]

num_batches = 12
days_per_batch = 1
dates = []

# Not starting at 1st of month
# BARRA loads monthly data as per UTC (i.e. starting at 01-01T00:00)
# Himawari daily data organised by AUD days (i.e. starts at 12-31T17:00)
# So starting at one can be annoying
# Taking days from the middle of the month avoids this
for day in range(2, 12):
    first = f'2025-01-{day:02d}'
    first_dt = datetime.strptime(first, "%Y-%m-%d")
    for x in range(num_batches):
        start_dt = first_dt + relativedelta(months =  x)
        start_date = start_dt.strftime("%Y-%m-%d")
        dates.append(start_date)

for date in dates:
    
    # Generate a unique file name based on iteration
    joboutdir = '/home/548/cd3022/repos/solar-nowcast/jobs/prepare_data/'
    job_script_filename = joboutdir + f'{script_to_run}___{date}___{days_per_batch}.qsub'
    
    # Open the file for writing
    with open(job_script_filename, "w") as f3:
        f3.write('#!/bin/bash \n')
        f3.write('#PBS -l walltime=1:00:00 \n')
        f3.write('#PBS -l mem=96GB \n')
        f3.write('#PBS -l ncpus=48 \n')
        f3.write('#PBS -l jobfs=10GB \n')
        f3.write('#PBS -l storage=gdata/dk92+gdata/rt52+scratch/nf33+gdata/rv74+gdata/rq0+gdata/ra22+gdata/xp65+gdata/er8+scratch/er8+gdata/ob53 \n')
        f3.write('#PBS -l other=hyperthread \n')
        f3.write('#PBS -q normal \n')
        f3.write('#PBS -P er8 \n')
        f3.write(f'#PBS -o /home/548/cd3022/repos/solar-nowcast/logs/prepare_data/{script_to_run}___{date}___{days_per_batch}.oe \n')
        f3.write('#PBS -j oe \n')
        f3.write('cd /home/548/cd3022/repos/solar-nowcast \n') 
        f3.write('source env.sh \n')
        f3.write(f'python3 mouse/{script_to_run}.py {date} {days_per_batch}\n')


    # Submit the generated script to the job scheduler (PBS) using qsub
    try:
        # Run the qsub command and submit the script
        subprocess.run(['qsub', job_script_filename], check=True)
        print(f"Job script {job_script_filename} submitted successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job script {job_script_filename}: {e}")