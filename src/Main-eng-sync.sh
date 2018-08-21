#!/bin/bash
# Usage: ./Main-eng-dynet.sh ResultsFolderName NMT_ENSEMBLES BEAM MODEL_TYPE
# Usage: ./Main-eng-dynet.sh eng 1 3 -nmt

###########################################
## POINTERS TO WORKING AND DATA DIRECTORIES
###########################################
#

export PF="eng"
export DIR=/home/tanja/uzh-corpuslab-normalization

export DATA=$DIR/data/canonical-segmentation/english
export EXTRADATA=/$DIR/data/canonical-segmentation/additional/eng/aspell.txt

export SCRIPTS=/home/tanja/Normalization-sgnmt/scripts-dynet-segmentation #TBC

export SEGM=/home/christof/Chintang/uzh-corpuslab-morphological-segmentation/SEGM

#LM paths
export LD_LIBRARY_PATH=/home/christof/Chintang/swig-srilm:$LD_LIBRARY_PATH
export PYTHONPATH=/home/christof/Chintang/swig-srilm:$PYTHONPATH
export PATH=/home/christof/Chintang/SRILM/bin:/home/christof/Chintang/SRILM/bin/i686-m64:$PATH
export MANPATH=/home/christof/Chintang/SRILM/man:$PATH

export PATH=/home/christof/Chintang/swig:$PATH

#MERT path
export MERT=/home/christof/Chintang/uzh-corpuslab-morphological-segmentation/zmert_v1.50

#Pretrained NMT model
export MODEL=/mnt/results/


export NMT_SEED=$2
export BEAM=$3

for (( n=0; n<=0; n++ )) #data split (from 0 till 9)

do

(
mkdir -p $DIR/results/{PF}/$n
export RESULTS_ALL=$DIR/results/{PF}/$n

export TRAINDATA=$DATA/train$n
export DEVDATA=$DATA/dev$n
export TESTDATA=$DATA/test$n

nmt_predictors="nmt"

mkdir -p $RESULTS_ALL/${NMT_SEED}
export RESULTS=$RESULTS_ALL/${NMT_SEED}
nmt_path="--nmt_path=$MODEL/${PF}_${n}_nmt_${NMT_SEED}" #temporary over nmt models

# Prepare target and source dictionaries
cp $MODEL/${PF}_${n}_nmt_1/vocab.txt $EXPER_DATA/vocab.trg
cp $MODEL/${PF}_${n}_nmt_1/vocab.txt $EXPER_DATA/vocab.src


#
###########################################
## PREPARATION - masking and vocabulary
###########################################
#

# Prepare train set (charcter based - add spaces)
cut -f1 $TRAINDATA > $RESULTS/train.src
cut -f2 $TRAINDATA > $RESULTS/train.trg

# Prepare test set (charcter based - add spaces)
cut -f1 $TESTDATA > $RESULTS/test.src
cut -f2 $TESTDATA > $RESULTS/test.trg

# Prepare validation set (charcter based - add spaces)
cut -f1 $DEVDATA > $RESULTS/dev.src
cut -f2 $DEVDATA > $RESULTS/dev.trg

##########################################
# TRAINING NMT
##########################################

### TO BE REPLACED WITH DYNET TRAINING
if [[ $4 == "-train" ]]; then # Train nmt models
echo "TO BE REPLACED WITH DYNET TRAINING"

############################################
# DECODING NMT + EVALUATION on dev and test
############################################

elif [[ $4 == "-nmt" ]]; then # Only evaluate ensembles of nmt models

echo "TO BE REPLACED WITH DYNET EVAL"

PYTHONIOENCODING=utf8 python2.7 $SCRIPTS/accuracy-det.py $DEVDATA $TRAINDATA $RESULTS/dev_out_vanilla.txt $RESULTS/dev.src $RESULTS/Errors_vanilla_dev.txt > $RESULTS/Accuracy_vanilla_dev_det.txt #TBC

# evaluate on tokens - detailed output
PYTHONIOENCODING=utf8 python2.7 $SCRIPTS/accuracy-det.py $TESTDATA $TRAINDATA $RESULTS/test_out_vanilla.txt $RESULTS/test.src $RESULTS/Errors_vanilla_test.txt > $RESULTS/Accuracy_vanilla_test_det.txt #TBC

else # nmt + LM

##########################################
# LM over words
##########################################

# Use extra data for language model over words
if [[ $4 == *"e"* ]]; then

# Prepare extended training target file
cut -f2 $EXTRADATA > $RESULTS/extra.train.trg
# Extend training set
cat $RESULTS/train.trg $RESULTS/extra.train.trg > $RESULTS/train_ext.trg
# train LM
(ngram-count -text $RESULTS/train_ext.trg -lm $RESULTS/morfs.lm -order 3 -write $RESULTS/morfs.lm.counts -kndiscount -interpolate ) || { echo "Backup to ukn "; (ngram-count -text $RESULTS/train_ext.trg -lm $RESULTS/morfs.lm -order 3 -write $RESULTS/morfs.lm.counts -ukndiscount -interpolate );} || { echo "Backup to wb "; (ngram-count -text $RESULTS/train_ext.trg -lm $RESULTS/morfs.lm -order 3 -write $RESULTS/morfs.lm.counts -wbdiscount -interpolate );}

else
# train LM
(ngram-count -text $RESULTS/train_ext.trg -lm $RESULTS/morfs.lm -order 3 -write $RESULTS/morfs.lm.counts -kndiscount -interpolate ) || { echo "Backup to ukn "; (ngram-count -text $RESULTS/train_ext.trg -lm $RESULTS/morfs.lm -order 3 -write $RESULTS/morfs.lm.counts -ukndiscount -interpolate );} || { echo "Backup to wb "; (ngram-count -text $RESULTS/train_ext.trg -lm $RESULTS/morfs.lm -order 3 -write $RESULTS/morfs.lm.counts -wbdiscount -interpolate );}

fi


##########################################
# LM over chars
##########################################
#
# Prepare extended training target file
cut -f2 $EXTRADATA > $RESULTS/extra.train.trg

# Prepare LM train file with chars masked by int using NMT target chars2int dictionary
python2.7 $SCRIPTS/apply_wmap.py -m $RESULTS/vocab.trg < $RESULTS/extra.train.trg > $RESULTS/extra.train.itrg

# train LM
(ngram-count -text $EXPER_DATA/extra.train.itrg -lm $EXPER_DATA/chars.lm -order 7 -write $EXPER_DATA/chars.lm.counts -kndiscount -interpolate  -gt3min 1 -gt4min 1 -gt5min 1 -gt6min 1 -gt7min 1 ) || { echo "Backup to ukn "; (ngram-count -text $EXPER_DATA/extra.train.itrg -lm $EXPER_DATA/chars.lm -order 7 -write $EXPER_DATA/chars.lm.counts -ukndiscount -interpolate  -gt3min 1 -gt4min 1 -gt5min 1 -gt6min 1 -gt7min 1);} || { echo "Backup to wb "; (ngram-count -text $EXPER_DATA/extra.train.itrg -lm $EXPER_DATA/chars.lm -order 7 -write $EXPER_DATA/chars.lm.counts -wbdiscount -interpolate  -gt3min 1 -gt4min 1 -gt5min 1 -gt6min 1 -gt7min 1 );}



##########################################
# MERT for NMT & LM + EVALUATION
##########################################

##To make sure:
# Change -r in ZMERT_cfg.txt to dev.trg


cp -R $MERT/segm $RESULTS/mert$4${PF}$n${NMT_SEED}

export MERTEXPER=$RESULTS/mert$4${PF}$n${NMT_SEED}


cd $MERTEXPER

# NMT + Language Model over chars
if [[ $4 == "-c" ]]; then
# passed to zmert: commands to decode n-best list from dev file
echo "python $SEGM/decode_segm.py --predictors $nmt_predictors,srilmchar --decoder syncbeam $nmt_path --srilmchar_path=$EXPER_DATA/chars.lm --srilmchar_order=7 --max_len_factor=3 --src_wmap=$EXPER_DATA/vocab.src --trg_wmap=$EXPER_DATA/vocab.trg --src_test=$EXPER_DATA/dev.src --outputs=nbest --nbest=0 --output_path=nbest.out --sync_symbol=$SyncSymbol --beam=$BEAM" > SDecoder_cmd

# passed to zmert: commands to decode 1-best list from test file
echo "python $SEGM/decode_segm.py --predictors $nmt_predictors,srilmchar --decoder syncbeam $nmt_path --srilmchar_path=$EXPER_DATA/chars.lm --srilmchar_order=7 --max_len_factor=3 --src_wmap=$EXPER_DATA/vocab.src --trg_wmap=$EXPER_DATA/vocab.trg --src_test=$EXPER_DATA/test.src --outputs=text --output_path=test.out --sync_symbol=$SyncSymbol --beam=$BEAM" > SDecoder_cmd_test

nmt_w=$(echo "scale=2;1/$NMT_ENSEMBLES" | bc)
#echo $nmt_w
while read num; do nmt_weights+="nmt$num $nmt_w\n"; done < <(seq $NMT_ENSEMBLES)
echo -e "cands_file=nbest.txt\ncands_per_sen=12\ntop_n=12\n\n${nmt_weights}lm 0.001" > SDecoder_cfg.txt

while read num; do nmt_params+="nmt$num\t|||\t${nmt_w}\tFix\t0\t+1\t0\t+1\n"; done < <(seq $NMT_ENSEMBLES)
echo -e "${nmt_params}lm\t|||\t0.001\tOpt\t0\t+Inf\t0\t+1\nnormalization = none" > params.txt



# NMT + Language Model over chars + length control
elif [[ $4 == "-cL" ]]; then

# passed to zmert: commands to decode n-best list from dev file
echo "python $SEGM/decode_segm.py --predictors $nmt_predictors,wc,srilm --decoder syncbeam $nmt_path --nmt_config src_vocab_size=$SrcVocab,trg_vocab_size=$TrgVocab --srilm_path=$EXPER_DATA/chars.lm --srilm_order=7 --max_len_factor=3 --src_wmap=$EXPER_DATA/vocab.src --trg_wmap=$EXPER_DATA/vocab.trg --src_test=$EXPER_DATA/dev.src --outputs=nbest --nbest=0 --output_path=nbest.out --sync_symbol=$SyncSymbol --beam=$BEAM" > SDecoder_cmd

# passed to zmert: commands to decode 1-best list from test file
echo "python $SEGM/decode_segm.py --predictors $nmt_predictors,wc,srilm --decoder syncbeam $nmt_path --nmt_config src_vocab_size=$SrcVocab,trg_vocab_size=$TrgVocab --srilm_path=$EXPER_DATA/chars.lm --srilm_order=7 --max_len_factor=3 --src_wmap=$EXPER_DATA/vocab.src --trg_wmap=$EXPER_DATA/vocab.trg --src_test=$EXPER_DATA/test.src --outputs=text --output_path=test.out --sync_symbol=$SyncSymbol --beam=$BEAM" > SDecoder_cmd_test

nmt_w=$(echo "scale=2;1/$NMT_ENSEMBLES" | bc)
#echo $nmt_w
while read num; do nmt_weights+="nmt$num $nmt_w\n"; done < <(seq $NMT_ENSEMBLES)
echo -e "cands_file=nbest.txt\ncands_per_sen=12\ntop_n=12\n\n${nmt_weights}wc 1.0\nlm 0.1" > SDecoder_cfg.txt

while read num; do nmt_params+="nmt$num\t|||\t${nmt_w}\tFix\t0\t+1\t0\t+1\n"; done < <(seq $NMT_ENSEMBLES)
echo -e "${nmt_params}wc\t|||\t1.0\tOpt\t0\t+Inf\t0\t+3\nlm\t|||\t0.1\tOpt\t0\t+Inf\t0\t+1\nnormalization = none" > params.txt


# NMT + Language Model over words
elif [[ $4 == "-w" ]] || [[ $4 == "-we" ]]; then
# passed to zmert: commands to decode n-best list from dev file
echo "python $SEGM/decode_segm.py --predictors $nmt_predictors,word2char_srilm --decoder syncbeam $nmt_path  --word2char_map=$EXPER_DATA/vocab.m2c --srilm_path=$EXPER_DATA/morfs.lm --srilm_order=3 --max_len_factor=3 --src_wmap=$EXPER_DATA/vocab.src --trg_wmap=$EXPER_DATA/vocab.trg --src_test=$EXPER_DATA/dev.src --outputs=nbest --nbest=0 --output_path=nbest.out --sync_symbol=$SyncSymbol --beam=$BEAM" > SDecoder_cmd

# passed to zmert: commands to decode 1-best list from test file
echo "python $SEGM/decode_segm.py --predictors $nmt_predictors,word2char_srilm --decoder syncbeam $nmt_path  --word2char_map=$EXPER_DATA/vocab.m2c --srilm_path=$EXPER_DATA/morfs.lm --srilm_order=3 --max_len_factor=3 --src_wmap=$EXPER_DATA/vocab.src --trg_wmap=$EXPER_DATA/vocab.trg --src_test=$EXPER_DATA/test.src --outputs=text --output_path=test.out --sync_symbol=$SyncSymbol --beam=$BEAM" > SDecoder_cmd_test

nmt_w=$(echo "scale=2;1/$NMT_ENSEMBLES" | bc)
#echo $nmt_w
while read num; do nmt_weights+="nmt$num $nmt_w\n"; done < <(seq $NMT_ENSEMBLES)
echo -e "cands_file=nbest.txt\ncands_per_sen=12\ntop_n=12\n\n${nmt_weights}lm 0.1" > SDecoder_cfg.txt

while read num; do nmt_params+="nmt$num\t|||\t${nmt_w}\tFix\t0\t+1\t0\t+1\n"; done < <(seq $NMT_ENSEMBLES)
echo -e "${nmt_params}lm\t|||\t0.1\tOpt\t0\t+Inf\t0\t+1\nnormalization = none" > params.txt


# NMT + Language Model over words  + length control
elif [[ $4 == "-wL" ]] || [[ $4 == "-weL" ]]; then

    # passed to zmert: commands to decode n-best list from dev file
    echo "python $SEGM/decode_segm.py --predictors $nmt_predictors,wc,word2char_srilm --decoder syncbeam $nmt_path --nmt_config src_vocab_size=$SrcVocab,trg_vocab_size=$TrgVocab --word2char_map=$EXPER_DATA/vocab.m2c --srilm_path=$EXPER_DATA/morfs.lm --srilm_order=3 --max_len_factor=3 --src_wmap=$EXPER_DATA/vocab.src --trg_wmap=$EXPER_DATA/vocab.trg --src_test=$EXPER_DATA/dev.src --outputs=nbest --nbest=0 --output_path=nbest.out --sync_symbol=$SyncSymbol --beam=$BEAM" > SDecoder_cmd

    # passed to zmert: commands to decode 1-best list from test file
    echo "python $SEGM/decode_segm.py --predictors $nmt_predictors,wc,word2char_srilm --decoder syncbeam $nmt_path --nmt_config src_vocab_size=$SrcVocab,trg_vocab_size=$TrgVocab --word2char_map=$EXPER_DATA/vocab.m2c --srilm_path=$EXPER_DATA/morfs.lm --srilm_order=3 --max_len_factor=3 --src_wmap=$EXPER_DATA/vocab.src --trg_wmap=$EXPER_DATA/vocab.trg --src_test=$EXPER_DATA/test.src --outputs=text --output_path=test.out --sync_symbol=$SyncSymbol --beam=$BEAM" > SDecoder_cmd_test

    nmt_w=$(echo "scale=2;1/$NMT_ENSEMBLES" | bc)
    #echo $nmt_w
    while read num; do nmt_weights+="nmt$num $nmt_w\n"; done < <(seq $NMT_ENSEMBLES)
    echo -e "cands_file=nbest.txt\ncands_per_sen=12\ntop_n=12\n\n${nmt_weights}wc 1.0\nlm 0.1" > SDecoder_cfg.txt

    while read num; do nmt_params+="nmt$num\t|||\t${nmt_w}\tFix\t0\t+1\t0\t+1\n"; done < <(seq $NMT_ENSEMBLES)
    echo -e "${nmt_params}wc\t|||\t1.0\tOpt\t0\t+Inf\t0\t+3\nlm\t|||\t0.1\tOpt\t0\t+Inf\t0\t+1\nnormalization = none" > params.txt


# NMT + Language Model over chars + Language Model over words
elif [[ $4 == "-cw" ]] || [[ $4 == "-cwe" ]]; then
# passed to zmert: commands to decode n-best list from dev file
echo "python $SEGM/decode_segm.py --predictors $nmt_predictors,word2char_srilm,srilmchar  --decoder syncbeam $nmt_path --nmt_config src_vocab_size=$SrcVocab,trg_vocab_size=$TrgVocab --word2char_map=$EXPER_DATA/vocab.m2c --srilmchar_path=$EXPER_DATA/chars.lm --srilmchar_order=7 --srilm_path=$EXPER_DATA/morfs.lm --srilm_order=3 --max_len_factor=3 --src_wmap=$EXPER_DATA/vocab.src --trg_wmap=$EXPER_DATA/vocab.trg --src_test=$EXPER_DATA/dev.src --outputs=nbest --nbest=0 --output_path=nbest.out --sync_symbol=$SyncSymbol --beam=$BEAM" > SDecoder_cmd

# passed to zmert: commands to decode 1-best list from test file
echo "python $SEGM/decode_segm.py --predictors $nmt_predictors,word2char_srilm,srilmchar --decoder syncbeam $nmt_path --nmt_config src_vocab_size=$SrcVocab,trg_vocab_size=$TrgVocab --word2char_map=$EXPER_DATA/vocab.m2c --srilmchar_path=$EXPER_DATA/chars.lm --srilmchar_order=7 --srilm_path=$EXPER_DATA/morfs.lm --srilm_order=3 --max_len_factor=3 --src_wmap=$EXPER_DATA/vocab.src --trg_wmap=$EXPER_DATA/vocab.trg --src_test=$EXPER_DATA/test.src --outputs=text --output_path=test.out --sync_symbol=$SyncSymbol --beam=$BEAM" > SDecoder_cmd_test

nmt_w=$(echo "scale=2;1/$NMT_ENSEMBLES" | bc)
#echo $nmt_w
while read num; do nmt_weights+="nmt$num $nmt_w\n"; done < <(seq $NMT_ENSEMBLES)
echo -e "cands_file=nbest.txt\ncands_per_sen=12\ntop_n=12\n\n${nmt_weights}lm1 0.1\nlm2 0.001" > SDecoder_cfg.txt

while read num; do nmt_params+="nmt$num\t|||\t${nmt_w}\tFix\t0\t+1\t0\t+1\n"; done < <(seq $NMT_ENSEMBLES)
echo -e "${nmt_params}lm1\t|||\t0.1\tOpt\t0\t+Inf\t0\t+1\nlm2\t|||\t0.001\tOpt\t0\t+Inf\t0\t+1\nnormalization = none" > params.txt


# NMT + Language Model over chars + Language Model over words  + length control
elif [[ $4 == "-cwL" ]] || [[ $4 == "-cweL" ]]; then
# passed to zmert: commands to decode n-best list from dev file
echo "python $SEGM/decode_segm.py --predictors $nmt_predictors,wc,word2char_srilm,srilmchar  --decoder syncbeam $nmt_path --nmt_config src_vocab_size=$SrcVocab,trg_vocab_size=$TrgVocab --word2char_map=$EXPER_DATA/vocab.m2c --srilmchar_path=$EXPER_DATA/chars.lm --srilmchar_order=7 --srilm_path=$EXPER_DATA/morfs.lm --srilm_order=3 --max_len_factor=3 --src_wmap=$EXPER_DATA/vocab.src --trg_wmap=$EXPER_DATA/vocab.trg --src_test=$EXPER_DATA/dev.src --outputs=nbest --nbest=0 --output_path=nbest.out --sync_symbol=$SyncSymbol --beam=$BEAM" > SDecoder_cmd

# passed to zmert: commands to decode 1-best list from test file
echo "python $SEGM/decode_segm.py --predictors $nmt_predictors,wc,word2char_srilm,srilmchar --decoder syncbeam $nmt_path --nmt_config src_vocab_size=$SrcVocab,trg_vocab_size=$TrgVocab --word2char_map=$EXPER_DATA/vocab.m2c --srilmchar_path=$EXPER_DATA/chars.lm --srilmchar_order=7 --srilm_path=$EXPER_DATA/morfs.lm --srilm_order=3 --max_len_factor=3 --src_wmap=$EXPER_DATA/vocab.src --trg_wmap=$EXPER_DATA/vocab.trg --src_test=$EXPER_DATA/test.src --outputs=text --output_path=test.out --sync_symbol=$SyncSymbol --beam=$BEAM" > SDecoder_cmd_test

nmt_w=$(echo "scale=2;1/$NMT_ENSEMBLES" | bc)
#echo $nmt_w
while read num; do nmt_weights+="nmt$num $nmt_w\n"; done < <(seq $NMT_ENSEMBLES)
echo -e "cands_file=nbest.txt\ncands_per_sen=12\ntop_n=12\n\n${nmt_weights}wc 1.0\nlm1 0.1\nlm2 0.001" > SDecoder_cfg.txt

while read num; do nmt_params+="nmt$num\t|||\t${nmt_w}\tFix\t0\t+1\t0\t+1\n"; done < <(seq $NMT_ENSEMBLES)
echo -e "${nmt_params}wc\t|||\t1.0\tOpt\t0\t+Inf\t0\t+3\nlm1\t|||\t0.1\tOpt\t0\t+Inf\t0\t+1\nlm2\t|||\t0.001\tOpt\t0\t+Inf\t0\t+1\nnormalization = none" > params.txt
fi


cp $RESULTS/dev.trg $MERTEXPER
cp $RESULTS/test.src $MERTEXPER

java -cp $MERT/lib/zmert.jar ZMERT -maxMem 500 ZMERT_cfg.txt

## copy test out file - for analysis
cp test.out $RESULTS/test_out_mert.txt
#
## copy n-best file for dev set with optimal weights - for analysis
cp nbest.out $RESULTS/nbest_dev_mert.out
#
cp SDecoder_cfg.txt.ZMERT.final $RESULTS/params-mert-ens.txt
#
##evaluate on tokens
PYTHONIOENCODING=utf8 python2.7 $SCRIPTS/accuracy-only.py $RESULTS/test_out_mert.txt  $EXPER_DATA/test.trg > $RESULTS/Accuracy_mert_test.txt #TBC
#
##evaluate on tokens - detailed output
PYTHONIOENCODING=utf8 python2.7 $SCRIPTS/accuracy-det.py $TESTDATA $TRAINDATA $RESULTS/test_out_mert.txt $EXPER_DATA/test.src $RESULTS/Errors_mert_test.txt > $RESULTS/Accuracy_mert_test_det.txt #TBC


#rm -r $MERTEXPER

fi


#rm -r $EXPER/exper_data

echo "Process {$n} finished"
)

done
