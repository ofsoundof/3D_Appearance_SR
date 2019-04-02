#! /bin/bash


export SUBSET=.
export EPOCH=100
for s in 4; do
    for n in one; do
        #if [ ! $s = 4 ] || [ ! $n = one ]; then
            for res in lr hr; do
                while true; do
                    job_string=$(qstat -u yawli)
                    sub_string='yawli'
                    diff=${job_string//${sub_string}}
                    count=$(((${#job_string} - ${#diff}) / ${#sub_string}))
                    echo "The number of running jobs is $count."
                    if [ 6 -gt $count ]; then
                        export SCALE=$s
                        export N_PATCH=$(($s * 48))
                        export MODEL_ONE=$n
                        export MODEL=FINETUNE
                        export NORMAL=$res
                        echo "The number of running jobs ${count} is smaller than 6."
                        echo "Submit one job for scale $SCALE, patch size $N_PATCH, model $MODEL, split ${MODEL_ONE}, resolution ${NORMAL}."
                        bash finetune_sr.sh
                        break
                    fi
                    echo "The number of running jobs is equal to 6."
                    t=900
                    echo "Wait for $t seconds."
                    sleep $t
                done
            done
        #fi
    done
done
