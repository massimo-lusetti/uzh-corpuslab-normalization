#!/bin/bash
# Usage Main-train.sh DataFolder ResultPrefix DataSplit NMTSeed/Ensemble
# ./Main-seg-soft-train.sh canonical-segmentation/english/ eng 0 1
# ./Main-seg-soft-train.sh canonical-segmentation/english/ eng 0 ens
# ./Main-seg-soft-train.sh canonical-segmentation/indonesian/ ind 0 1
# ./Main-seg-soft-train.sh canonical-segmentation/indonesian/ ger 0 1
##########################################################################################

export DATA=$1

export n=$3
export k=$4

#for (( n=0; n<=4; n++ ))
#do
#(
export TRAIN=$DATA/train$n
export DEV=$DATA/dev$n
export TEST=$DATA/test$n

export PR=$2_$n
echo "$PR"

if [[ $k != 'ens' ]]; then
########### train + eval of individual models
#for (( k=1; k<=5; k++ ))
#do
#(
PYTHONIOENCODING=utf8 python norm_soft.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${PR}_nmt_$k  --epochs=30 --input_format=0,2

wait

PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$DEV --beam=3 --pred_path=best.dev.3 --input_format=0,2 &
PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$TEST --beam=3 --pred_path=best.test.3 --input_format=0,2
#) &
#done
#
#wait

else
########### Evaluate NMT ensemble 5

PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3,${PR}_nmt_4,${PR}_nmt_5 --test_path=$DEV --beam=3 --pred_path=best.dev.3 ${PR}_nmt_ens5 --input_format=0,2 &
PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3,${PR}_nmt_4,${PR}_nmt_5 --test_path=$TEST --beam=3 --pred_path=best.test.3 ${PR}_nmt_ens5 --input_format=0,2

#)
#done

fi
