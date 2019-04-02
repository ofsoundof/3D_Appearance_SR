#! /bin/bash


# run NLR and NLR-Sub
for d in ETH3D; do
for s in 4 3 2; do
    for n in one two; do

        for m in FINETUNE; do
            while true; do
            
            	# another method to count the number of submitted jobs.
                count=$(qstat -u yawli | tail -n +3 | wc -l)
                echo "The number of running jobs is $count."
                
                # submit jobs
                if [ 6 -gt $count ]; then
                    export SCALE=$s
                    export N_PATCH=$(($s * 48))
                    export MODEL_ONE=$n
                    export MODEL=$m
                    export SUBMODEL=NLR
                    export SUBSET=$d
                    export EPOCH=100
                    echo "The number of running jobs ${count} is smaller than 6."
                    echo "Submit one job for scale $SCALE, patch size $N_PATCH, epoch $EPOCH, model $MODEL, set split ${MODEL_ONE}, subset ${SUBSET}."
                    bash finetune_sr.sh
                    break
                fi
                
                # wait for some time
                echo "The number of running jobs is equal to 6."
                echo "Current time is $(date +%Y-%m-%d/%l:%M:%S)"
                t=450
                echo "Wait for $t seconds."
                sleep $t
            done
        done

    done
done
done
