#!/bin/bash

#$ -l rt_G.large=1              
#$ -l h_rt=72:00:00              
#$ -j y                           
#$ -cwd                        

source /etc/profile.d/modules.sh
module load singularitypro
singularity exec --nv --bind /groups/gcd50654/tier4/dataset/semantickitti/dataset:$HOME/sourceset \
                      --bind /groups/gcd50654/tier4/dataset/vls:$HOME/targetset salsanext_step1.sif \
                      bash sin.sh