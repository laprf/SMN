#!/usr/bin/bash

BACKBONE="resnet18"  # resnet18, swin_t, pvt_v2_b1
DATASET="HS-SOD" # HSOD-BIT, HS-SOD
DATAPATH="DataStorage/$DATASET"
GTPATH="$DATAPATH/GT/test"

if [ $BACKBONE != "pvt_v2_b0" ] && [ $BACKBONE != "resnet18" ] && [ $BACKBONE != "swin_t" ] && [ $BACKBONE != "pvt_v2_b1" ]; then
    echo "BACKBONE must be in 'pvt_v2_b0', 'resnet18', 'swin_t' or 'pvt_v2_b1'"
    exit 1
fi

# train
 CUDA_VISIBLE_DEVICES="0" python train.py --backbone=$BACKBONE --data_path=$DATAPATH --dataset=$DATASET

# test
CUDA_VISIBLE_DEVICES="0" python test.py --backbone=$BACKBONE --model_path="DataStorage/$DATASET/checkpoints/$BACKBONE/model-best" --data_path=$DATAPATH --dataset=$DATASET

# eval
CUDA_VISIBLE_DEVICES="0" python eval/main.py --sm_dir="$DATAPATH/exp_results/$BACKBONE/test_result" --gt_dir=$GTPATH --datasets=$DATASET
CUDA_VISIBLE_DEVICES="0" python eval/eval_AUC_CC.py --sm_dir="$DATAPATH/exp_results/$BACKBONE/test_result" --gt_dir=$GTPATH