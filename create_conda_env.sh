#!/bin/bash

# shellcheck disable=SC1091
source ~/miniconda3/bin/activate
conda create -y -n waterstart --file conda_requirements.txt -c pytorch -c conda-forge
