#!/bin/bash
# Usage Main-sms-soft-train-pos.sh model_file_name data_folder use_aux_loss
# ./Main-sms-soft-train-pos.sh norm_soft SMS/POS
# ./Main-sms-soft-train-pos.sh norm_soft_pos SMS/POS
# ./Main-sms-soft-train-pos.sh norm_soft_context SMS/POS
# ./Main-sms-soft-train-pos.sh norm_soft_context SMS/POS aux
##########################################################################################


export TRAIN=$2/train.txt
export DEV=$2/dev.txt
export TEST=$2/test.txt

export MODEL=$1
if [[ $3 == "aux" ]]; then
export PR="sms_aux"
else
export PR="sms"
fi
echo "$PR"

########### SEED 1 + eval
if [[ $3 == "aux" ]]; then
PYTHONIOENCODING=utf8 python ${MODEL}.py train --dynet-seed 1 --train_path=$TRAIN --dev_path=$DEV ${PR}_${MODEL}_1  --epochs=30 --lowercase --aux_pos_task
elif [[ $2 == "norm_soft_pos" ]]; then
PYTHONIOENCODING=utf8 python ${MODEL}.py train --dynet-seed 1 --train_path=$TRAIN --dev_path=$DEV ${PR}_${MODEL}_1  --epochs=30 --lowercase --pos_split_space
else
PYTHONIOENCODING=utf8 python ${MODEL}.py train --dynet-seed 1 --train_path=$TRAIN --dev_path=$DEV ${PR}_${MODEL}_1  --epochs=30 --lowercase
fi

PYTHONIOENCODING=utf8 python ${MODEL}.py test ${PR}_${MODEL}_1 --test_path=$DEV --beam=3 --pred_path=best.dev.3  --lowercase &

PYTHONIOENCODING=utf8 python ${MODEL}.py test ${PR}_${MODEL}_1 --test_path=$TEST --beam=3 --pred_path=best.test.3  --lowercase

########### SEED >1 + eval
## the vocabulary of SEED 0 is used for other models in ensemble
#for (( k=2; k<=5; k++ ))
#do
#(
#if [[ $3 == "aux" ]]; then
#PYTHONIOENCODING=utf8 python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${PR}_${MODEL}_$k  --epochs=30 --lowercase --aux_pos_task --vocab_path=${PR}_${MODEL}_1/vocab.txt
#elif [[ $2 == "norm_soft_pos" ]]; then
#PYTHONIOENCODING=utf8 python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${PR}_${MODEL}_$k  --epochs=30 --lowercase --pos_split_space --vocab_path=${PR}_${MODEL}_1/vocab.txt
#else
#PYTHONIOENCODING=utf8 python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${PR}_${MODEL}_$k  --epochs=30 --lowercase --vocab_path=${PR}_${MODEL}_1/vocab.txt
#fi

##
#PYTHONIOENCODING=utf8 python ${MODEL}.py test ${PR}_${MODEL}_$k --test_path=$DEV --beam=3 --pred_path=best.dev.3  --lowercase &
#PYTHONIOENCODING=utf8 python ${MODEL}.py test ${PR}_${MODEL}_$k --test_path=$TEST --beam=3 --pred_path=best.test.3  --lowercase
#)
#done


########### Evaluate ensemble 5

#PYTHONIOENCODING=utf8 python ${MODEL}.py ensemble_test ${PR}_${MODEL}_1,${PR}_${MODEL}_2,${PR}_${MODEL}_3,${PR}_${MODEL}_4,${PR}_${MODEL}_5 --test_path=$DEV --beam=3 --pred_path=best.dev.3 ${PR}_${MODEL}_ens5  --lowercase &
#PYTHONIOENCODING=utf8 python ${MODEL}.py ensemble_test ${PR}_${MODEL}_1,${PR}_${MODEL}_2,${PR}_${MODEL}_3,${PR}_${MODEL}_4,${PR}_${MODEL}_5 --test_path=$TEST --beam=3 --pred_path=best.test.3 ${PR}_${MODEL}_ens5  --lowercase &

