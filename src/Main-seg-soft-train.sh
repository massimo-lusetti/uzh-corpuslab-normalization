#!/bin/bash
# Usage Main-train.sh DataFolder ResultPrefix
# ./Main-seg-soft-train.sh canonical-segmentation/english/ eng
# ./Main-seg-soft-train.sh canonical-segmentation/indonesian/ ind
# ./Main-seg-soft-train.sh canonical-segmentation/indonesian/ ger
##########################################################################################

export DATA=$1

for (( n=0; n<=4; n++ ))
do
(
export TRAIN=$DATA/train$n
export DEV=$DATA/dev$n
export TEST=$DATA/test$n

export PR=$2_$n
echo "$PR"

########### SEED 1
PYTHONIOENCODING=utf8 python norm_soft.py train --dynet-seed 1 --train_path=$TRAIN --dev_path=$DEV ${PR}_nmt_1  --epochs=30 --input_format=0,2

#evaluation
PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_1 --test_path=$DEV --beam=1 --pred_path=best.dev.1 --input_format=0,2
PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_1 --test_path=$DEV --beam=3 --pred_path=best.dev.3 --input_format=0,2
PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_1 --test_path=$TEST --beam=1 --pred_path=best.test.1 --input_format=0,2
PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_1 --test_path=$TEST --beam=3 --pred_path=best.test.3 --input_format=0,2


########### SEED >1 + eval
for (( k=2; k<=5; k++ ))
do
(
PYTHONIOENCODING=utf8 python norm_soft.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${PR}_nmt_$k  --epochs=30 --vocab_path=${PR}_nmt_1/vocab.txt --input_format=0,2

PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$DEV --beam=1 --pred_path=best.dev.1 --input_format=0,2
PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$DEV --beam=3 --pred_path=best.dev.3 --input_format=0,2
PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$TEST --beam=1 --pred_path=best.test.1 --input_format=0,2
PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$TEST --beam=3 --pred_path=best.test.3 --input_format=0,2
) &
done


########### Evaluate NMT

for (( k=1; k<=5; k++ ))
do
(
PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$DEV --beam=1 --pred_path=best.dev.1 --input_format=0,2 &
PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$DEV --beam=3 --pred_path=best.dev.3 --input_format=0,2 &
PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$TEST --beam=1 --pred_path=best.test.1 --input_format=0,2 &
PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$TEST --beam=3 --pred_path=best.test.3 --input_format=0,2

) &
done

########### Evaluate NMT ensemble 3

PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3 --test_path=$DEV --beam=1 --pred_path=best.dev.1 ${PR}_nmt_ens3 --input_format=0,2
PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3 --test_path=$DEV --beam=3 --pred_path=best.dev.3 ${PR}_nmt_ens3 --input_format=0,2
PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3 --test_path=$TEST --beam=1 --pred_path=best.test.1 ${PR}_nmt_ens3 --input_format=0,2
PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3 --test_path=$TEST --beam=3 --pred_path=best.test.3 ${PR}_nmt_ens3 --input_format=0,2

########### Evaluate NMT ensemble 5

PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3,${PR}_nmt_4,${PR}_nmt_5 --test_path=$DEV --beam=1 --pred_path=best.dev.1 ${PR}_nmt_ens5 --input_format=0,2
PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3,${PR}_nmt_4,${PR}_nmt_5 --test_path=$DEV --beam=3 --pred_path=best.dev.3 ${PR}_nmt_ens5 --input_format=0,2
PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3,${PR}_nmt_4,${PR}_nmt_5 --test_path=$TEST --beam=1 --pred_path=best.test.1 ${PR}_nmt_ens5 --input_format=0,2
PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3,${PR}_nmt_4,${PR}_nmt_5 --test_path=$TEST --beam=3 --pred_path=best.test.3 ${PR}_nmt_ens5 --input_format=0,2

)
done

