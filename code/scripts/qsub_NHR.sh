#! /bin/bash


# run NHR
export SUBSET=.
export EPOCH=100
for s in 2 3 4; do
    for n in one two; do
        for normal in hr; do
            while true; do
            	# count the number of submitted jobs.
                job_string=$(qstat -u yawli)
                sub_string='yawli'
                diff=${job_string//${sub_string}}
                count=$(((${#job_string} - ${#diff}) / ${#sub_string}))
                echo "The number of running jobs is $count."
                
                # qsub jobs
                if [ 6 -gt $count ]; then
                    export SCALE=$s
                    export N_PATCH=$(($s * 48))
                    export MODEL_ONE=$n
                    export MODEL=FINETUNE
                    export SUBMODEL=NHR
                    export NORMAL=$normal
                    echo "The number of running jobs ${count} is smaller than 6."
                    echo "Submit one job for scale $SCALE, patch size $N_PATCH, model $MODEL, split ${MODEL_ONE}, resolution ${NORMAL}."
                    bash finetune_sr.sh
                    break
                fi
                echo "The number of running jobs is equal to 6."
                
                # wait for some time
                t=900
                echo "Wait for $t seconds."
                sleep $t
            done
        done
    done
done
