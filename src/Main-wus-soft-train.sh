#!/bin/bash
# Usage Main-train-archi.sh
# ./Main-train-archi.sh
##########################################################################################


#export TRAIN=/home/tanja/Normalization-dynet/data/wus/nolinks-emoji-rpl/train.txt
#export DEV=/home/tanja/Normalization-dynet/data/wus/nolinks-emoji-rpl/tune.txt
#export TEST=/home/tanja/Normalization-dynet/data/wus/nolinks-emoji-rpl/test.txt
#
#export PR="wusmod"
#echo "$PR"


export TRAIN=/home/tanja/normalisation_input/wus/train.txt
export DEV=/home/tanja/normalisation_input/wus/tune.txt
export TEST=/home/tanja/normalisation_input/wus/test.txt

export PR="wusinit"
echo "$PR"

############ SEED 1 + eval
#PYTHONIOENCODING=utf8 python norm_soft.py train --dynet-seed 1 --train_path=$TRAIN --dev_path=$DEV ${PR}_nmt_1  --epochs=30  
##
#
#PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_1 --test_path=$DEV --beam=1 --pred_path=best.dev.1  
#PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_1 --test_path=$DEV --beam=3 --pred_path=best.dev.3  
#PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_1 --test_path=$TEST --beam=1 --pred_path=best.test.1  
#PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_1 --test_path=$TEST --beam=3 --pred_path=best.test.3  



############ SEED >1 + eval
#for (( k=2; k<=5; k++ ))
#do
#(
#PYTHONIOENCODING=utf8 python norm_soft.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${PR}_nmt_$k  --epochs=20 --vocab_path=${PR}_nmt_1/vocab.txt  
#
#PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$DEV --beam=1 --pred_path=best.dev.1  
#PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$DEV --beam=3 --pred_path=best.dev.3  
#PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$TEST --beam=1 --pred_path=best.test.1  
#PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$TEST --beam=3 --pred_path=best.test.3  
#)
#done



############ Evaluate NMT

#for (( k=1; k<=5; k++ ))
#do
#(
#PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$DEV --beam=1 --pred_path=best.dev.1
#PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$DEV --beam=3 --pred_path=best.dev.3
#PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$TEST --beam=1 --pred_path=best.test.1
#PYTHONIOENCODING=utf8 python norm_soft.py test ${PR}_nmt_$k --test_path=$TEST --beam=3 --pred_path=best.test.3  
#)
#done


############ Evaluate NMT ensemble 5

#PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3,${PR}_nmt_4,${PR}_nmt_5 --test_path=$DEV --beam=1 --pred_path=best.dev.1 ${PR}_nmt_ens5
#PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3,${PR}_nmt_4,${PR}_nmt_5 --test_path=$DEV --beam=3 --pred_path=best.dev.3 ${PR}_nmt_ens5
#PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3,${PR}_nmt_4,${PR}_nmt_5 --test_path=$TEST --beam=1 --pred_path=best.test.1 ${PR}_nmt_ens5
#PYTHONIOENCODING=utf8 python norm_soft.py ensemble_test ${PR}_nmt_1,${PR}_nmt_2,${PR}_nmt_3,${PR}_nmt_4,${PR}_nmt_5 --test_path=$TEST --beam=3 --pred_path=best.test.3 ${PR}_nmt_ens5



############ Evaluate NMT ensemble 5 from output

predout="../results/${PR}_nmt_ens5/best.dev.1.ens5.predictions"
pred="best.dev.1.predictions"
PYTHONIOENCODING=utf8 python  ensemble_from_output.py ../results/${PR}_nmt_1/$pred ../results/${PR}_nmt_2/$pred ../results/${PR}_nmt_3/$pred ../results/${PR}_nmt_4/$pred ../results/${PR}_nmt_5/$pred > $predout
PYTHONIOENCODING=utf8 python accuracy.py $predout  $DEV > ../results/${PR}_nmt_ens5/best.dev.1.ens5.eval

predout="../results/${PR}_nmt_ens5/best.dev.3.ens5.predictions"
pred="best.dev.3.predictions"
PYTHONIOENCODING=utf8 python  ensemble_from_output.py ../results/${PR}_nmt_1/$pred ../results/${PR}_nmt_2/$pred ../results/${PR}_nmt_3/$pred ../results/${PR}_nmt_4/$pred ../results/${PR}_nmt_5/$pred > $predout
PYTHONIOENCODING=utf8 python accuracy.py $predout  $DEV > ../results/${PR}_nmt_ens5/best.dev.3.ens5.eval

predout="../results/${PR}_nmt_ens5/best.test.1.ens5.predictions"
pred="best.test.1.predictions"
PYTHONIOENCODING=utf8 python  ensemble_from_output.py ../results/${PR}_nmt_1/$pred ../results/${PR}_nmt_2/$pred ../results/${PR}_nmt_3/$pred ../results/${PR}_nmt_4/$pred ../results/${PR}_nmt_5/$pred > $predout
PYTHONIOENCODING=utf8 python accuracy.py $predout  $TEST > ../results/${PR}_nmt_ens5/best.test.1.ens5.eval

predout="../results/${PR}_nmt_ens5/best.test.3.ens5.predictions"
pred="best.test.3.predictions"
PYTHONIOENCODING=utf8 python  ensemble_from_output.py ../results/${PR}_nmt_1/$pred ../results/${PR}_nmt_2/$pred ../results/${PR}_nmt_3/$pred ../results/${PR}_nmt_4/$pred ../results/${PR}_nmt_5/$pred > $predout
PYTHONIOENCODING=utf8 python accuracy.py $predout  $TEST > ../results/${PR}_nmt_ens5/best.test.3.ens5.eval


