#!/bin/bash
# Usage Main-archi-soft-train.sh
# ./Main-archi-soft-train.sh
##########################################################################################


export TRAIN=Archimob/train_archi.tsv
export DEV=Archimob/dev_archi.tsv
export TEST=Archimob/test_archi.tsv

export PR="arch"
echo "$PR"

########### SEED 1 + eval
PYTHONIOENCODING=utf8 python norm_soft.py train --dynet-seed 1 --train_path=$TRAIN --dev_path=$DEV ${PR}_nmt_1  --epochs=30 --input_format=0,2
#
PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_1 --test_path=$DEV --beam=1 --pred_path=best.dev.1 --input_format=0,2
PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_1 --test_path=$DEV --beam=3 --pred_path=best.dev.3 --input_format=0,2
PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_1 --test_path=$TEST --beam=1 --pred_path=best.test.1 --input_format=0,2
PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_1 --test_path=$TEST --beam=3 --pred_path=best.test.3 --input_format=0,2


########### SEED >1 + eval
## the vocabulary of SEED 0 is used for other models in ensemble
for (( k=2; k<=5; k++ ))
do
(
PYTHONIOENCODING=utf8 python norm_soft.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${PR}_nmt_$k  --epochs=20 --vocab_path=${PR}_nmt_1/vocab.txt --input_format=0,2
#
PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$DEV --beam=1 --pred_path=best.dev.1 --input_format=0,2
PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$DEV --beam=3 --pred_path=best.dev.3 --input_format=0,2
PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$TEST --beam=1 --pred_path=best.test.1 --input_format=0,2
PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$TEST --beam=3 --pred_path=best.test.3 --input_format=0,2
)
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

PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3 --test_path=$DEV --beam=1 --pred_path=best.dev.1 ${PR}_nmt_ens3 --input_format=0,2 &
PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3 --test_path=$DEV --beam=3 --pred_path=best.dev.3 ${PR}_nmt_ens3 --input_format=0,2 &
PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3 --test_path=$TEST --beam=1 --pred_path=best.test.1 ${PR}_nmt_ens3 --input_format=0,2 &
PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3 --test_path=$TEST --beam=3 --pred_path=best.test.3 ${PR}_nmt_ens3 --input_format=0,2 &

########### Evaluate NMT ensemble 5

PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3,${PR}_nmt_4,${PR}_nmt_5 --test_path=$DEV --beam=1 --pred_path=best.dev.1 ${PR}_nmt_ens5 --input_format=0,2 &
PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3,${PR}_nmt_4,${PR}_nmt_5 --test_path=$DEV --beam=3 --pred_path=best.dev.3 ${PR}_nmt_ens5 --input_format=0,2 &
PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3,${PR}_nmt_4,${PR}_nmt_5 --test_path=$TEST --beam=1 --pred_path=best.test.1 ${PR}_nmt_ens5 --input_format=0,2 &
PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3,${PR}_nmt_4,${PR}_nmt_5 --test_path=$TEST --beam=3 --pred_path=best.test.3 ${PR}_nmt_ens5 --input_format=0,2 &

