#!/bin/bash

GPUS="0 1 2 3"
NUM_THREADS=3
MAPPING=/mnt/chotot/classifiers/human_mapping.txt
CLASSIFIER=/mnt/chotot/classifiers/transfer_classifier_epochs_500_batch_2048_ratios_0.8_0.1_0.1_learning_rate_0.0001_dropout_0.5_hidden_size_2048.pb
REDIS=10.0.0.177

while [[ $# -gt 1 ]]
do
key="$1"

case $key in
    -g|--gpus)
    GPUS="$2"
    shift # past argument
    ;;
    -t|--threads_per_gpu)
    NUM_THREADS="$2"
    shift # past argument
    ;;
    -m|--mapping)
    MAPPING="$2"
    shift # past argument
    ;;
    -c|--classifier)
    CLASSIFIER="$2"
    shift # past argument
    ;;
    -r|--redis)
    REDIS="$2"
    shift # past argument
    ;;
    --default)
    DEFAULT=YES
    ;;
    *)
            # unknown option
    ;;
esac
shift # past argument or value
done

for i in $GPUS; do # for gpus 0, 1, 2 and 3
    for j in `seq 1 $NUM_THREADS` ; do # start NUM_THREADS workers
	CUDA_VISIBLE_DEVICES=$i python transfer_classifier.py --mapping $MAPPING --classifier $CLASSIFIER --redis_server $REDIS &
    done
done
