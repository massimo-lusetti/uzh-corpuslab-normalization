#!/bin/bash
# Usage Main-sms-soft-train-pos.sh
# ./Main-sms-soft-train-pos.sh norm_soft
# ./Main-sms-soft-train-pos.sh norm_soft_pos
# ./Main-sms-soft-train-pos.sh norm_soft_context
##########################################################################################


export TRAIN=SMS/POS/train.txt
export DEV=SMS/POS/dev.txt
export TEST=SMS/POS/test.txt

export MODEL=$1
export PR="sms"
echo "$PR"

########### SEED 1 + eval
#PYTHONIOENCODING=utf8 python ${MODEL}.py train --dynet-seed 11 --train_path=$TRAIN --dev_path=$DEV ${PR}_${MODEL}_1  --epochs=30 --lowercase
#
PYTHONIOENCODING=utf8 python ${MODEL}.py test ${PR}_${MODEL}_1 --test_path=$DEV --beam=3 --pred_path=best.dev.3  --lowercase

#PYTHONIOENCODING=utf8 python ${MODEL}.py test ${PR}_${MODEL}_1 --test_path=$DEV --beam=1 --pred_path=best.dev.1  --lowercase
#PYTHONIOENCODING=utf8 python ${MODEL}.py test ${PR}_${MODEL}_1 --test_path=$DEV --beam=3 --pred_path=best.dev.3  --lowercase
#PYTHONIOENCODING=utf8 python ${MODEL}.py test ${PR}_${MODEL}_1 --test_path=$TEST --beam=1 --pred_path=best.test.1  --lowercase
#PYTHONIOENCODING=utf8 python ${MODEL}.py test ${PR}_${MODEL}_1 --test_path=$TEST --beam=3 --pred_path=best.test.3  --lowercase


########### SEED >1 + eval
## the vocabulary of SEED 0 is used for other models in ensemble
#for (( k=2; k<=5; k++ ))
#do
#(
#PYTHONIOENCODING=utf8 python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${PR}_${MODEL}_$k  --epochs=20 --vocab_path=${PR}_${MODEL}_1/vocab.txt  --lowercase
##
#PYTHONIOENCODING=utf8 python ${MODEL}.py test ${PR}_${MODEL}_$k --test_path=$DEV --beam=1 --pred_path=best.dev.1  --lowercase
#PYTHONIOENCODING=utf8 python ${MODEL}.py test ${PR}_${MODEL}_$k --test_path=$DEV --beam=3 --pred_path=best.dev.3  --lowercase
#PYTHONIOENCODING=utf8 python ${MODEL}.py test ${PR}_${MODEL}_$k --test_path=$TEST --beam=1 --pred_path=best.test.1  --lowercase
#PYTHONIOENCODING=utf8 python ${MODEL}.py test ${PR}_${MODEL}_$k --test_path=$TEST --beam=3 --pred_path=best.test.3  --lowercase
#)
#done


########### Evaluate NMT

#for (( k=1; k<=5; k++ ))
#do
#(
#PYTHONIOENCODING=utf8 python ${MODEL}.py test ${PR}_${MODEL}_$k --test_path=$DEV --beam=1 --pred_path=best.dev.1  --lowercase &
#PYTHONIOENCODING=utf8 python ${MODEL}.py test ${PR}_${MODEL}_$k --test_path=$DEV --beam=3 --pred_path=best.dev.3  --lowercase &
#PYTHONIOENCODING=utf8 python ${MODEL}.py test ${PR}_${MODEL}_$k --test_path=$TEST --beam=1 --pred_path=best.test.1  --lowercase&
#PYTHONIOENCODING=utf8 python ${MODEL}.py test ${PR}_${MODEL}_$k --test_path=$TEST --beam=3 --pred_path=best.test.3  --lowercase
#) &
#done

########### Evaluate NMT ensemble 3

#PYTHONIOENCODING=utf8 python ${MODEL}.py ensemble_test ${PR}_${MODEL}_1,${PR}_${MODEL}_2,${PR}_${MODEL}_3 --test_path=$DEV --beam=1 --pred_path=best.dev.1 ${PR}_${MODEL}_ens3  --lowercase &
#PYTHONIOENCODING=utf8 python ${MODEL}.py ensemble_test ${PR}_${MODEL}_1,${PR}_${MODEL}_2,${PR}_${MODEL}_3 --test_path=$DEV --beam=3 --pred_path=best.dev.3 ${PR}_${MODEL}_ens3  --lowercase &
#PYTHONIOENCODING=utf8 python ${MODEL}.py ensemble_test ${PR}_${MODEL}_1,${PR}_${MODEL}_2,${PR}_${MODEL}_3 --test_path=$TEST --beam=1 --pred_path=best.test.1 ${PR}_${MODEL}_ens3   --lowercase&
#PYTHONIOENCODING=utf8 python ${MODEL}.py ensemble_test ${PR}_${MODEL}_1,${PR}_${MODEL}_2,${PR}_${MODEL}_3 --test_path=$TEST --beam=3 --pred_path=best.test.3 ${PR}_${MODEL}_ens3  --lowercase &

########### Evaluate NMT ensemble 5

#PYTHONIOENCODING=utf8 python ${MODEL}.py ensemble_test ${PR}_${MODEL}_1,${PR}_${MODEL}_2,${PR}_${MODEL}_3,${PR}_${MODEL}_4,${PR}_${MODEL}_5 --test_path=$DEV --beam=1 --pred_path=best.dev.1 ${PR}_${MODEL}_ens5  --lowercase &
#PYTHONIOENCODING=utf8 python ${MODEL}.py ensemble_test ${PR}_${MODEL}_1,${PR}_${MODEL}_2,${PR}_${MODEL}_3,${PR}_${MODEL}_4,${PR}_${MODEL}_5 --test_path=$DEV --beam=3 --pred_path=best.dev.3 ${PR}_${MODEL}_ens5  --lowercase &
#PYTHONIOENCODING=utf8 python ${MODEL}.py ensemble_test ${PR}_${MODEL}_1,${PR}_${MODEL}_2,${PR}_${MODEL}_3,${PR}_${MODEL}_4,${PR}_${MODEL}_5 --test_path=$TEST --beam=1 --pred_path=best.test.1 ${PR}_${MODEL}_ens5   --lowercase &
#PYTHONIOENCODING=utf8 python ${MODEL}.py ensemble_test ${PR}_${MODEL}_1,${PR}_${MODEL}_2,${PR}_${MODEL}_3,${PR}_${MODEL}_4,${PR}_${MODEL}_5 --test_path=$TEST --beam=3 --pred_path=best.test.3 ${PR}_${MODEL}_ens5  --lowercase &

