#! /bin/bash -f
ll=loglike.txt

#nvidia-smi --query-gpu=index,timestamp,name,driver_version,pstate,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv,nounits --filename=gpu-log.csv
#sed -i 's/,\s/\t/g' gpu-log.csv
#less -S gpu-log.csv | awk -F'\t' '$9 < 16000 {print $9}' > gpu_id.txt

s1=(AE AC BE BN BW FW)
s2=(EN ES EW FN FE FW)

for i in ${s1[@]}
do

    echo "------------------------------------------------------"
    echo "          RUNNING DATASET ${i}                        "
    echo "------------------------------------------------------"
    cd parent/
    ./main_parent start ../../IC/series1/Control_s1_${i} Control_s1_${i}
    cd ../
    ./main ../best_params_new/final_${i}s1_best.txt Control_s1_${i} 1
    cd parent/
    ./main_parent stop ../../IC/series1/Control_s1_${i} Control_s1_${i}
    cd ../
    make clear
done

for i in ${s2[@]}
do

    echo "------------------------------------------------------"
    echo "          RUNNING DATASET ${i}                        "
    echo "------------------------------------------------------"
    cd parent/
    ./main_parent start ../../IC/series2/Control_s2_${i} Control_s2_${i}
    cd ../
    ./main ../best_params_new/final_${i}s2_best.txt Control_s2_${i} 1
    cd parent/
    ./main_parent stop ../../IC/series2/Control_s2_${i} Control_s2_${i}
    cd ../
    make clear
done

