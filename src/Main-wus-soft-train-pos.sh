#!/bin/bash
# Usage Main-wus-soft-train-pos.sh model_file_name data_folder use_aux_loss_and/or_pos_feat
# ./Main-wus-soft-train-pos.sh norm_soft wus/phase2/btagger
# ./Main-wus-soft-train-pos.sh norm_soft_pos wus/phase2/btagger
# ./Main-wus-soft-train-pos.sh norm_soft_context wus/phase2/btagger
# ./Main-wus-soft-train-pos.sh norm_soft_context wus/phase2/btagger pos
# ./Main-wus-soft-train-pos.sh norm_soft_context wus/phase2/btagger aux
# ./Main-wus-soft-train-pos.sh norm_soft_context wus/phase2/btagger pos_aux
##########################################################################################


export TRAIN=$2/train_silverpos.txt
export DEV=$2/dev_autopos.txt
export TEST=$2/test_autopos.txt

export MODEL=$1
if [[ $3 == "aux" ]]; then
export PR="wus_aux"
elif [[ $3 == "pos" ]]; then
export PR="wus_pos"
elif [[ $3 == "pos_aux" ]]; then
export PR="wus_pos_aux"
else
export PR="wus"
fi
echo "$PR"

########### SEED 1 + eval
if [[ $3 == "aux" ]]; then
PYTHONIOENCODING=utf8 python ${MODEL}.py train --dynet-seed 1 --train_path=$TRAIN --dev_path=$DEV ${PR}_${MODEL}_1  --epochs=40 --lowercase --aux_pos_task
elif [[ $3 == "pos" ]]; then
PYTHONIOENCODING=utf8 python ${MODEL}.py train --dynet-seed 1 --train_path=$TRAIN --dev_path=$DEV ${PR}_${MODEL}_1  --epochs=40 --lowercase --pos_feature
elif [[ $3 == "pos_aux" ]]; then
PYTHONIOENCODING=utf8 python ${MODEL}.py train --dynet-seed 1 --train_path=$TRAIN --dev_path=$DEV ${PR}_${MODEL}_1  --epochs=40 --lowercase --pos_feature --aux_pos_task

elif [[ $1 == "norm_soft_pos" ]]; then
PYTHONIOENCODING=utf8 python ${MODEL}.py train --dynet-seed 1 --train_path=$TRAIN --dev_path=$DEV ${PR}_${MODEL}_1  --epochs=40 --lowercase
else
PYTHONIOENCODING=utf8 python ${MODEL}.py train --dynet-seed 1 --train_path=$TRAIN --dev_path=$DEV ${PR}_${MODEL}_1  --epochs=40 --lowercase
fi

PYTHONIOENCODING=utf8 python ${MODEL}.py test ${PR}_${MODEL}_1 --test_path=$DEV --beam=3 --pred_path=best.dev.3  --lowercase &

PYTHONIOENCODING=utf8 python ${MODEL}.py test ${PR}_${MODEL}_1 --test_path=$TEST --beam=3 --pred_path=best.test.3  --lowercase

############ SEED >1 + eval
### the vocabulary of SEED 1 is used for other models in ensemble
for (( k=2; k<=5; k++ ))
do
(
if [[ $3 == "aux" ]]; then
PYTHONIOENCODING=utf8 python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${PR}_${MODEL}_$k  --epochs=40 --lowercase --aux_pos_task --char_vocab_path=${PR}_${MODEL}_1/char_vocab.txt --word_vocab_path=${PR}_${MODEL}_1/word_vocab.txt --feat_vocab_path=${PR}_${MODEL}_1/feat_vocab.txt
elif [[ $3 == "pos" ]]; then
PYTHONIOENCODING=utf8 python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${PR}_${MODEL}_$k  --epochs=40 --lowercase --pos_feature --char_vocab_path=${PR}_${MODEL}_1/char_vocab.txt --word_vocab_path=${PR}_${MODEL}_1/word_vocab.txt --feat_vocab_path=${PR}_${MODEL}_1/feat_vocab.txt
elif [[ $3 == "pos_aux" ]]; then
PYTHONIOENCODING=utf8 python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${PR}_${MODEL}_$k  --epochs=40 --lowercase --pos_feature --aux_pos_task --char_vocab_path=${PR}_${MODEL}_1/char_vocab.txt --word_vocab_path=${PR}_${MODEL}_1/word_vocab.txt --feat_vocab_path=${PR}_${MODEL}_1/feat_vocab.txt

elif [[ $1 == "norm_soft_pos" ]]; then
PYTHONIOENCODING=utf8 python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${PR}_${MODEL}_$k  --epochs=40 --lowercase --vocab_path=${PR}_${MODEL}_1/vocab.txt
else
PYTHONIOENCODING=utf8 python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${PR}_${MODEL}_$k  --epochs=40 --lowercase --char_vocab_path=${PR}_${MODEL}_1/char_vocab.txt --word_vocab_path=${PR}_${MODEL}_1/word_vocab.txt --feat_vocab_path=${PR}_${MODEL}_1/feat_vocab.txt
fi

PYTHONIOENCODING=utf8 python ${MODEL}.py test ${PR}_${MODEL}_$k --test_path=$DEV --beam=3 --pred_path=best.dev.3  --lowercase &
PYTHONIOENCODING=utf8 python ${MODEL}.py test ${PR}_${MODEL}_$k --test_path=$TEST --beam=3 --pred_path=best.test.3  --lowercase
) &
done


############ Evaluate ensemble 5

PYTHONIOENCODING=utf8 python ${MODEL}.py ensemble_test ${PR}_${MODEL}_1,${PR}_${MODEL}_2,${PR}_${MODEL}_3,${PR}_${MODEL}_4,${PR}_${MODEL}_5 --test_path=$DEV --beam=3 --pred_path=best.dev.3 ${PR}_${MODEL}_ens5  --lowercase
PYTHONIOENCODING=utf8 python ${MODEL}.py ensemble_test ${PR}_${MODEL}_1,${PR}_${MODEL}_2,${PR}_${MODEL}_3,${PR}_${MODEL}_4,${PR}_${MODEL}_5 --test_path=$TEST --beam=3 --pred_path=best.test.3 ${PR}_${MODEL}_ens5  --lowercase

