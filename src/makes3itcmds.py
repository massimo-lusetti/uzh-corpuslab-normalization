#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

RESULTS_NAME = ('x-d{DATASET}-'
               'n{HIDDEN}_{LAYERS}-w{INPUT}-'
               'e{EPOCHS}_{PATIENCE}-o{OPTIMIZATION}-x')

CALL = """{CMDPREFIX} python norm_soft.py --dynet-seed {SEED} --dynet-mem {MEM} \
 --input={INPUT} --hidden={HIDDEN}  \
 --layers={LAYERS}  --optimization={OPTIMIZATION} \
 --patience={PATIENCE} --epochs={EPOCHS} \
 {TRAINPATH} {DEVPATH} {RESULTSPATH} --test_path={TESTPATH} \
  2>&1 > {OUTPATH}  && touch {DONEPATH}
"""

RESULTSDIR = '../results'

CMDPREFIX = 'PYTHONIOENCODING=utf8'

DATASET_PATH = '/home/tanja/Normalization-dynet/data/wus'
#DATASETs = ['init', 'nolinks',  'nolinks-emoji-rpl']
DATASETs = ['init', 'nolinks-emoji-rpl']
DATASET_TRAINFILE_NAME = 'train.txt'
DATASET_DEVFILE_NAME = 'tune.txt'
DATASET_TESTFILE_NAME = 'test.txt'

SEED = 1
MEM = 1024
INPUTs = [100, 200, 300]
HIDDENs = [100, 200, 300]
LAYERSs = [1, 2]
OPTIMIZATIONs = ['SGD', 'ADADELTA']
PATIENCE = 10
EPOCHS = 30


#calls = []
for INPUT in INPUTs:
    for HIDDEN in HIDDENs:
        for LAYERS in LAYERSs:
            for OPTIMIZATION in OPTIMIZATIONs:
                for DATASET in DATASETs:
                    RESULTSPATH = os.path.join(RESULTSDIR,
                                               RESULTS_NAME.format(DATASET=DATASET,
                                                                   HIDDEN=HIDDEN,
                                                                   LAYERS=LAYERS,
                                                                   INPUT=INPUT,
                                                                   EPOCHS=EPOCHS,
                                                                   PATIENCE=PATIENCE,
                                                                   OPTIMIZATION=OPTIMIZATION))

                    OUTPATH = os.path.join(RESULTSPATH, 'output.stdout')
                    DONEPATH = os.path.join(RESULTSPATH, 'model.done')
                    
                    TRAINPATH = os.path.join(DATASET_PATH, DATASET, DATASET_TRAINFILE_NAME)
                    DEVPATH   = os.path.join(DATASET_PATH, DATASET, DATASET_DEVFILE_NAME)
                    TESTPATH  = os.path.join(DATASET_PATH, DATASET, DATASET_TESTFILE_NAME)


                    call = CALL.format(SEED=SEED,
                                       MEM=MEM,
                                       TRAINPATH=TRAINPATH,
                                       HIDDEN=HIDDEN,
                                       LAYERS=LAYERS,
                                       INPUT=INPUT,
                                       EPOCHS=EPOCHS,
                                       PATIENCE=PATIENCE,
                                       OPTIMIZATION=OPTIMIZATION,
                                       DEVPATH=DEVPATH,
                                       TESTPATH=TESTPATH,
                                       RESULTSPATH=RESULTSPATH,
                                       OUTPATH=OUTPATH,
                                       DONEPATH=DONEPATH,
                                       CMDPREFIX=CMDPREFIX)
                    call = 'cd $CONLL_HOME/src && mkdir -p {} && {}'.format(RESULTSPATH, call)
                    print  call
#                    calls.append(call)




