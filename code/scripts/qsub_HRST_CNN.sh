#! /bin/bash


# run HRST-CNN
export EPOCH=100
export INPUT_RES=hr
for subset in .;
	do
	export SUBSET=$subset
	for s in 2 4; do
		for n in one two; do

            while true; do
            	# count the number of submitted jobs.
                job_string=$(qstat -u yawli)
                sub_string='yawli'
                diff=${job_string//${sub_string}}
                count=$(((${#job_string} - ${#diff}) / ${#sub_string}))
                echo "The number of running jobs is $count."
                #num_eth3d=$(ls /home/yawli/Documents/3d-appearance-benchmark/SR/texture/ETH3D/x${s}/*.png | wc -l)
                
                # qsub jobs
                if [ 6 -gt $count ]; then
                    export SCALE=$s
                    export N_PATCH=$(($s * 48))
                    export MODEL_ONE=$n
                    export MODEL=FINETUNE
                    export SUBMODEL=HRST_CNN
                    echo "The number of running jobs ${count} is smaller than 6."
                    echo "Submit one job for scale $SCALE, patch size $N_PATCH, model $MODEL, split ${MODEL_ONE}, resolution ${NORMAL}, subset ${SUBSET}."
                    bash finetune_sr.sh
                    break
                fi
                echo "The number of running jobs is equal to 6."
                
                # wait for some time
                t=300
                echo "Wait for $t seconds."
                sleep $t
            done

		done
	done
done
