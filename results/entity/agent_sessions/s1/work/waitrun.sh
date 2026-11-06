#!/bin/bash
# wait until GPU has room, then run the given python script
cd /home/ubuntu/wp/runs/entity/agent/s1
while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 8000 ]; do sleep 10; done
echo "GPU free at $(date)"
PYTHONIOENCODING=utf-8 /opt/pytorch/bin/python "$@" 2>&1 | grep -v -i warning
echo "DONE exit"
