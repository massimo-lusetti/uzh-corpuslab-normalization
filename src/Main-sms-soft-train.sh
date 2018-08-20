#!/bin/bash
# Usage Main-sms-soft-train.sh
# ./Main-sms-soft-train.sh
##########################################################################################


export TRAIN=SMS/POS/train.txt
export DEV=SMS/POS/dev.txt
export TEST=SMS/POS/test.txt

export PR="sms"
echo "$PR"

########### SEED 1 + eval
PYTHONIOENCODING=utf8 python norm_soft.py train --dynet-seed 1 --train_path=$TRAIN --dev_path=$DEV ${PR}_nmt_1  --epochs=30 --lowercase
#
PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_1 --test_path=$DEV --beam=1 --pred_path=best.dev.1  --lowercase
PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_1 --test_path=$DEV --beam=3 --pred_path=best.dev.3  --lowercase
PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_1 --test_path=$TEST --beam=1 --pred_path=best.test.1  --lowercase
PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_1 --test_path=$TEST --beam=3 --pred_path=best.test.3  --lowercase


########### SEED >1 + eval
## the vocabulary of SEED 0 is used for other models in ensemble
#for (( k=2; k<=5; k++ ))
#do
#(
#PYTHONIOENCODING=utf8 python norm_soft.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${PR}_nmt_$k  --epochs=20 --vocab_path=${PR}_nmt_1/vocab.txt  --lowercase
##
#PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$DEV --beam=1 --pred_path=best.dev.1  --lowercase
#PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$DEV --beam=3 --pred_path=best.dev.3  --lowercase
#PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$TEST --beam=1 --pred_path=best.test.1  --lowercase
#PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$TEST --beam=3 --pred_path=best.test.3  --lowercase
#)
#done


########### Evaluate NMT

#for (( k=1; k<=5; k++ ))
#do
#(
#PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$DEV --beam=1 --pred_path=best.dev.1  --lowercase &
#PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$DEV --beam=3 --pred_path=best.dev.3  --lowercase &
#PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$TEST --beam=1 --pred_path=best.test.1  --lowercase&
#PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$TEST --beam=3 --pred_path=best.test.3  --lowercase
#) &
#done

########### Evaluate NMT ensemble 3

#PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3 --test_path=$DEV --beam=1 --pred_path=best.dev.1 ${PR}_nmt_ens3  --lowercase &
#PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3 --test_path=$DEV --beam=3 --pred_path=best.dev.3 ${PR}_nmt_ens3  --lowercase &
#PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3 --test_path=$TEST --beam=1 --pred_path=best.test.1 ${PR}_nmt_ens3   --lowercase&
#PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3 --test_path=$TEST --beam=3 --pred_path=best.test.3 ${PR}_nmt_ens3  --lowercase &

########### Evaluate NMT ensemble 5

#PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3,${PR}_nmt_4,${PR}_nmt_5 --test_path=$DEV --beam=1 --pred_path=best.dev.1 ${PR}_nmt_ens5  --lowercase &
#PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3,${PR}_nmt_4,${PR}_nmt_5 --test_path=$DEV --beam=3 --pred_path=best.dev.3 ${PR}_nmt_ens5  --lowercase &
#PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3,${PR}_nmt_4,${PR}_nmt_5 --test_path=$TEST --beam=1 --pred_path=best.test.1 ${PR}_nmt_ens5   --lowercase &
#PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3,${PR}_nmt_4,${PR}_nmt_5 --test_path=$TEST --beam=3 --pred_path=best.test.3 ${PR}_nmt_ens5  --lowercase &

