#! /bin/bash
#ll=loglike.txt

#nvidia-smi --query-gpu=index,timestamp,name,driver_version,pstate,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv,nounits --filename=gpu-log.csv
#sed -i 's/,\s/\t/g' gpu-log.csv
#less -S gpu-log.csv | awk -F'\t' '$9 < 16000 {print $9}' > gpu_id.txt

./main params.txt Control_s1_AC 16
