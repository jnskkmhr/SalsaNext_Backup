#!/bin/bash

conda init && \
source ~/.bashrc
source /opt/anaconda3/bin/activate salsanext
source ~/SalsaNext/train.sh -d ~/dataset -a ~/SalsaNext/salsanext.yml -l ~/SalsaNext/log_pi -c 0,1,2,3