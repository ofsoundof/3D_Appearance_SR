export SGE_GPU_ALL="$(ls -rt /tmp/lock-gpu*/info.txt | xargs grep -h  $(whoami) | awk '{print $2}' | paste -sd "," -)"
export SGE_GPU=$(echo $SGE_GPU_ALL |rev|cut -d, -f1|rev) # USE LAST GPU by request time
echo "SGE gpu=$SGE_GPU_ALL available"
echo "SGE gpu=$SGE_GPU allocated in this use"
export CUDA_VISIBLE_DEVICES=$SGE_GPU
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
