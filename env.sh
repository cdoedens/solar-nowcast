#!/bin/bash

# load analysis3 conda environment
module use /g/data/xp65/public/modules
module load conda/analysis3

# root directory for this repo
export ROOT=/home/548/${USER}/repos/solar-nowcast
export MODULES=${ROOT}/modules

# append python path
export PYTHONPATH=${MODULES}:${PYTHONPATH}