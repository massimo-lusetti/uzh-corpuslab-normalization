#!/usr/bin/env python
# -*- coding: utf-8 -*
""" This file evaluates results with details over seen/unseen segments(morphemes/words)

Usage:
  accuracy-det.py eval [--input_format=INPUT_FORMAT] [--lowercase=LOW] [--extended_train_data=EXT_FILE]
  TRAIN_DATA TEST_DATA PREDICTIONS RESULTS_FILE RESULTS_ERRORS_FILE
  accuracy-det.py eval_baseline [--input_format=INPUT_FORMAT] [--lowercase=LOW] [--pred_file=PRED_FILE]
  TRAIN_DATA TEST_DATA
  accuracy-det.py eval_ambiguity [--input_format=INPUT_FORMAT] [--lowercase=LOW]
  TRAIN_DATA TEST_DATA PREDICTIONS RESULTS_FILE RESULTS_ERRORS_FILE
  

Arguments:
TRAIN_DATA          train file path
TEST_DATA           test file path
PREDICTIONS         path for the predictions for the test data
RESULTS_FILE        path to save the evaluation
RESULTS_ERRORS_FILE path to save errors

Options:
  -h --help                         show this help message and exit
  --input_format=INPUT_FORMAT       coma-separated list of input, output, features columns [default: 0,1]
  --lowercase=LOW                   use lowercased data [default: True]
  --extended_train_data=EXT_FILE    extended data used for LM training, one-column format
  --pred_file=PRED_FILE             file to write the results of baseline evaluation
"""

from __future__ import division
__author__ = 'Tanya'
from docopt import docopt
import codecs
import unicodedata
from collections import defaultdict



def evaluate_baseline(trainin,gold,input_format,lowercase=False, pred_f=None):
    train_lexicon_m = defaultdict(int)
    test_lexicon_m = defaultdict(int)
    train_lexicon_w = defaultdict(int)
    test_lexicon_w = defaultdict(int)
    test_dict = {} # test_word: dict(predict:freq)
    train_dict = {} # test_word: dict(predict:freq)
    
    input = input_format[0]
    pred = input_format[1]
    
    # Read lexicon of training set
    trainin_f = codecs.open(trainin,'r','utf-8')
    for i,line in enumerate(trainin_f):
        #if i < 40:
        #        print line
        if len(line.strip()) != 0:
            line = line.strip().lower() if lowercase else line.strip()
            lineitems = line.split('\t')
            word = lineitems[input]
            segm = lineitems[pred]
            train_lexicon_w[word] += 1
            morfs = lineitems[pred].split(' ')
            for m in morfs:
                train_lexicon_m[m] += 1
            if not word in train_dict.keys():
                train_dict[word] = {}
                train_dict[word][segm]=1
            else:
                if segm not in train_dict[word].keys():
                    train_dict[word][segm]=1
                else:
                    train_dict[word][segm]+=1

    # Read lexicon of test set
    gold_f = codecs.open(gold,'r','utf-8')
    for i,line in enumerate(gold_f):
        #        if i < 40:
        if len(line.strip()) != 0:
            #            print line
            line = line.strip().lower() if lowercase else line.strip()
            lineitems = line.split('\t')
            word = lineitems[input]
            segm = lineitems[pred]
            test_lexicon_w[word] += 1
            morfs = lineitems[pred].split(' ')
            for m in morfs:
                test_lexicon_m[m] += 1
            if not word in test_dict.keys():
                test_dict[word] = {}
                test_dict[word][segm]=1
            else:
                if segm not in test_dict[word].keys():
                    test_dict[word][segm]=1
                else:
                    test_dict[word][segm]+=1

    amb_segm_test_candidates = {k:v for k,v in test_dict.items() if k in train_dict.keys()} #the values are test frequencies - for statistics
    amb_segm_train = {k:train_dict[k] for k,v in amb_segm_test_candidates.items() if len(train_dict[k])>1} # the values are train frequencies - for prediction
    amb_segm_test = {k:v for k,v in amb_segm_test_candidates.items() if len(train_dict[k])>1} # the values are test frequencies - for statistics
    print amb_segm_test.items()[:10]
    amb_segm_test_freq = {k:sum(v.values()) for k,v in amb_segm_test.items()}
    amb = sum(amb_segm_test_freq.values())
    corr_amb = 0 # number of correct ambigous

    amb_segm_test_tie_candidates = {k:v.values() for k,v in amb_segm_train.items()}
    amb_segm_test_tie_check = {k:v for k,v in amb_segm_test_tie_candidates.items() if v.count(v[0]) == len(v)}
    print amb_segm_test_tie_check.items()
    amb_segm_test_tie = {k:sum(v) for k,v in amb_segm_test_tie_candidates.items() if v.count(v[0]) == len(v)}
    amb_tie = sum(amb_segm_test_tie.values())
    corr_amb_tie = 0 # number of correct ambigous with tie
    amb_notie = amb - amb_tie
    corr_amb_notie = 0 # number of correct ambigous with tie


    
    not_amb_segm_test = {k:v for k,v in test_dict.items() if k not in amb_segm_test.keys()}
    seen_freq = {k:v.values()[0] for k,v in not_amb_segm_test.items() if k in train_lexicon_w.keys()}
    seen = sum(seen_freq.values())
    corr_seen = 0 # number of correct seen words
    
    unseen_freq = {k:v.values()[0] for k,v in not_amb_segm_test.items() if not k in train_lexicon_w.keys()}
    unseen = sum(unseen_freq.values())
    
    unseen_m_freq = {k:v.values()[0] for k,v in not_amb_segm_test.items() if ( not k in train_lexicon_w.keys() and not all(m in train_lexicon_m.keys() for m in v.keys()[0].split(' ')) )}
    unseen_m = sum(unseen_m_freq.values())
    corr_unseen_m = 0 # number of correct unseen words - new morphs
    
    unseen_new_comb = unseen - unseen_m
    corr_unseen_comb = 0 # number of correct unseen words - new combinations
    allc = 0
    corr = 0

    if pred_f:
        pred_f = codecs.open(pred_f,'w','utf-8')
        pred_f.write(u"{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format("input","gold","baseline", "ambig?", "ambig tie?", "new?", "unique?"))

    # baseline statistics
    gold_f.seek(0)
    for i,line in enumerate(gold_f):
        #if i < 5:
        if len(line.strip()) !=0:
            try:
                line = line.strip().lower() if lowercase else line.strip()
                lineitems = line.split('\t')
                w = lineitems[input]
                w_segm = lineitems[pred]
                lines = str(i)
                w_segm_morfs = w_segm.split(' ')
            
            except:
                print i, line
            allc +=1
            
            # seen and ambigous
            if w in amb_segm_test.keys():
                w_preds = amb_segm_train[w]
                w_baseline_pred = max(w_preds.keys(), key=lambda k: w_preds[k])
                if w_baseline_pred == w_segm:
                    corr +=1
                    corr_amb +=1
                    if w in amb_segm_test_tie.keys():
                        corr_amb_tie +=1
                    else:
                        corr_amb_notie +=1
            else:
                #new
                if w not in train_lexicon_w.keys():
                    w_baseline_pred = w
                    # new - old morphemes but new combination
                    if all(m in train_lexicon_m.keys() for m in w_segm_morfs):
                        if w_baseline_pred == w_segm:
                            corr +=1
                            corr_unseen_comb +=1
                    else:
                        # new - new morphemes
                        if w_baseline_pred == w_segm:
                            corr +=1
                            corr_unseen_m +=1
                #seen and unique
                else:
                    w_baseline_pred = test_dict[w].keys()[0]
                    if w_baseline_pred == w_segm:
                        corr +=1
                        corr_seen +=1
            if pred_f:
                w_new,w_unique,w_amb_tie,w_amb = False, False, False, False
                if w in amb_segm_test.keys():
                    w_amb = True
                    if w in amb_segm_test_tie.keys():
                        w_amb_tie = True
                else:
                    if w not in train_lexicon_w.keys():
                        w_new = True
                    else:
                        w_unique = True

                pred_f.write(u"{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(w,w_segm,w_baseline_pred, w_amb,w_amb_tie, w_new, w_unique))
                        
    print "\nDATA:"
    print "\n   TRAIN  TEST:"
    print "\n# of target segment tokens %d  %d" % (sum(train_lexicon_m.values()), sum(test_lexicon_m.values()))
    print "\n# of source word tokens %d  %d" % (sum(train_lexicon_w.values()), sum(test_lexicon_w.values()))
    print "\n# of ambigous source word tokens        %d (%.2f%%)" % (amb, float(amb)/float(sum(test_lexicon_w.values()))*100)
    print "\n# of ambigous source word tokens - ties        %d (%.2f%%)" % (amb_tie, float(amb_tie)/float(sum(test_lexicon_w.values()))*100)
    print "\n# of ambigous source word tokens - no ties        %d (%.2f%%)" % (amb_notie, float(amb_notie)/float(sum(test_lexicon_w.values()))*100)
    print "\n# of seen source word tokens        %d (%.2f%%)" % (seen, float(seen)/float(sum(test_lexicon_w.values()))*100)
    print "\n# of unseen source word tokens      %d (%.2f%%)" % (unseen, float(unseen)/float(sum(test_lexicon_w.values()))*100)
    print "\n# of unseen source word tokens - new target segments       %d (%.2f%%)" % (unseen_m, float(unseen_m)/float(sum(test_lexicon_w.values()))*100)
    print "\n# of unseen word tokens - new combination        %d (%.2f%%)" % (unseen_new_comb, float(unseen_new_comb)/float(sum(test_lexicon_w.values()))*100)

    print "\nPERFORMANCE:"
    print "\n        Number of predictions total: %d" % allc
    print "\nNumber of correct predictions total: %d (%.2f%%)" % (corr, float(corr)/float(allc)*100)
    print "\n                         - ambigous: %d (%.2f%%)" % (corr_amb, float(corr_amb)/float(amb)*100)
    print "\n                   - ambigous(ties): %d (%.2f%%)" % (corr_amb_tie, float(corr_amb_tie)/float(amb_tie)*100)
    print "\n                - ambigous(no ties): %d (%.2f%%)" % (corr_amb_notie, float(corr_amb_notie)/float(amb_notie)*100)
    if seen !=0:
        print "\n                   - seen words: %d (%.2f%%)" % (corr_seen, float(corr_seen)/float(seen)*100)
    print "\n            - unseen (new morphemes): %d (%.2f%%)" % (corr_unseen_m, float(corr_unseen_m)/float(unseen_m)*100)
    print "\n            - unseen (new combination): %d (%.2f%%)" % (corr_unseen_comb, float(corr_unseen_comb)/float(unseen_new_comb)*100)

def evaluate_ambiguity(trainin,gold,predict,file_out,file_out_errors,input_format,lowercase=False):
    train_lexicon_m = defaultdict(int)
    test_lexicon_m = defaultdict(int)
    train_lexicon_w = defaultdict(int)
    test_lexicon_w = defaultdict(int)
    test_dict = {} # test_word: dict(predict:freq)
    
    input = input_format[0]
    pred = input_format[1]
    pos_col = input_format[2]
    
    # Read lexicon of training set
    trainin_f = codecs.open(trainin,'r','utf-8')
    for i,line in enumerate(trainin_f):
        #if i < 40:
        #        print line
        if len(line.strip()) != 0:
            line = line.strip().lower() if lowercase else line.strip()
            lineitems = line.split('\t')
            word = lineitems[input]
            segm = lineitems[pred]
            train_lexicon_w[word] += 1
            morfs = lineitems[pred].split(' ')
            for m in morfs:
                train_lexicon_m[m] += 1

    # Read lexicon of test set
    gold_f = codecs.open(gold,'r','utf-8')
    for i,line in enumerate(gold_f):
        #        if i < 40:
        if len(line.strip()) != 0:
            #            print line
            line = line.strip().lower() if lowercase else line.strip()
            lineitems = line.split('\t')
            word = lineitems[input]
            segm = lineitems[pred]
            pos = lineitems[pos_col]
            test_lexicon_w[word] += 1
            morfs = lineitems[pred].split(' ')
            for m in morfs:
                test_lexicon_m[m] += 1
            if not word in test_dict.keys():
                test_dict[word] = {}
                test_dict[word][segm]={}
                test_dict[word][segm][pos] = 1
            else:
                if segm not in test_dict[word].keys():
                    test_dict[word][segm]={}
                    test_dict[word][segm][pos]=1
                else:
                    if pos not in test_dict[word][segm].keys():
                        test_dict[word][segm][pos]=1
                    else:
                        test_dict[word][segm][pos]+=1

    # Collect predictions
    predict_f = codecs.open(predict,'r','utf-8')
    pred_dict_ext = {}
    for j, line in enumerate(predict_f):
        line = line.strip().lower() if lowercase else line.strip()
        w, w_segm = line.split('\t')
        pred_dict_ext[(w,j+1)] = w_segm


    amb_segm_test = {k:v for k,v in test_dict.items() if len(v)>1}
    amb_segm_test_freq = {}
    pos_disamb_freq = {}
    for w,w_v in amb_segm_test.items():
        pos_sets = []
        freq = 0
        for seg,seg_v in w_v.items():
            pos_sets.append(set(seg_v.keys()))
            freq += sum(seg_v.values())
        amb_segm_test_freq[w] =freq
        u = set.intersection(*pos_sets)
        if len(u) == 0:
            pos_disamb_freq[w] = freq
    amb = sum(amb_segm_test_freq.values())
    pos_disamb = sum(pos_disamb_freq.values())

    errors = {}
    corr = 0
    corr_amb = 0 # number of correct ambigous
    corr_pos_disamb = 0 # number of correct ambigous which can be disambiguated with pos
    allc = 0

    gold_f.seek(0)
    for i,line in enumerate(gold_f):
        #if i < 5:
        if len(line.strip()) !=0:
            try:
                line = line.strip().lower() if lowercase else line.strip()
                lineitems = line.split('\t')
                w = lineitems[input]
                w_segm = lineitems[pred]
                lines = str(i)
                w_segm_morfs = w_segm.split(' ')
                
            except:
                print i, line

            allc += 1

            if pred_dict_ext[(w,allc)] == w_segm:
                corr += 1
        
                if w in amb_segm_test_freq.keys():
                    corr_amb += 1

                    if w in pos_disamb_freq.keys():
                        corr_pos_disamb +=1

            else:
    
                if (w,pred_dict_ext[(w,allc)], w_segm) not in errors.keys():
                    errors[(w,pred_dict_ext[(w,allc)], w_segm)] = [lines]
                else:
                    errors[(w,pred_dict_ext[(w,allc)], w_segm)].append(lines)
                        

    print "\nDATA:"
    print "\n# of ambigous source word tokens        %d (%.2f%%)" % (amb, amb/sum(test_lexicon_w.values())*100)
    print "\n# can be POS disambiguated:        %d (%.2f%%)" % (pos_disamb, pos_disamb/amb*100)

    print "\nPERFORMANCE:"
    print "\n        Number of predictions total: %d" % allc
    print "\nNumber of correct predictions total: %d (%.2f%%)" % (corr, corr/allc*100)
    print "\n                         - ambigous: %d (%.2f%%)" % (corr_amb, corr_amb/amb*100)
    print "\n                  - POS disambigous: %d (%.2f%%)" % (corr_pos_disamb, corr_pos_disamb/pos_disamb*100)

    with codecs.open(file_out_errors,'w','utf-8') as f:
    #f.write("\n\nERRORS:\n")
        f.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format("word","prediction", "gold", "err_freq", "ambigous?", "can be POS disamb?", "lines(test)"))
        orderd_w = sorted(errors.keys(), key=lambda v: v[1], reverse=True)
        for (w,pred,true_pred) in orderd_w:
            amb_type = w in amb_segm_test_freq.keys()
            pos_disamb_type = w in pos_disamb_freq.keys()
            f.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(w, pred, true_pred, len(errors[(w,pred,true_pred)]), amb_type, pos_disamb_type, ", ".join(errors[(w,pred,true_pred)])))


def evaluate(trainin,gold,predict,file_out,file_out_errors,input_format,lowercase=False, ext_trainin=None):
    train_lexicon_m = defaultdict(int)
    test_lexicon_m = defaultdict(int)
    train_lexicon_w = defaultdict(int)
    test_lexicon_w = defaultdict(int)
    test_dict = {} # test_word: dict(predict:freq)

    input = input_format[0]
    pred = input_format[1]
                 
    # Read lexicon of training set
    trainin_f = codecs.open(trainin,'r','utf-8')
    for i,line in enumerate(trainin_f):
        #if i < 40:
#        print line
        if len(line.strip()) != 0:
            line = line.strip().lower() if lowercase else line.strip()
            lineitems = line.split('\t')
            word = lineitems[input]
            segm = lineitems[pred]
            train_lexicon_w[word] += 1
            morfs = lineitems[pred].split(' ')
            for m in morfs:
                train_lexicon_m[m] += 1

    if ext_trainin: # extra train data for LM training, one-colum
        trainin_ext_f = codecs.open(ext_trainin,'r','utf-8')
        for i,line in enumerate(trainin_ext_f):
            if len(line.strip()) != 0:
                line = line.strip().lower() if lowercase else line.strip()
                morfs = line.split(' ')
                for m in morfs:
                    train_lexicon_m[m] += 1
    

    # Read lexicon of test set
    gold_f = codecs.open(gold,'r','utf-8')
    for i,line in enumerate(gold_f):
#        if i < 40:
        if len(line.strip()) != 0:
#            print line
            line = line.strip().lower() if lowercase else line.strip()
            lineitems = line.split('\t')
            word = lineitems[input]
            segm = lineitems[pred]
            test_lexicon_w[word] += 1
            morfs = lineitems[pred].split(' ')
            for m in morfs:
                test_lexicon_m[m] += 1
            if not word in test_dict.keys():
                test_dict[word] = {}
                test_dict[word][segm]=1
            else:
                if segm not in test_dict[word].keys():
                    test_dict[word][segm]=1
                else:
                    test_dict[word][segm]+=1

    errors = {}

        
    #LM Evaluation
    allc = 0 # total number of predictions  (that is the number of words in the input (gold))
    corr = 0 # total number of correct predictions
    
    
#    segm_test = {k:sum(v.values()) for k,v in test_dict.items()}
#    print sum(segm_test.values())
    amb_segm_test = {k:v for k,v in test_dict.items() if len(v)>1}
    amb_segm_test_freq = {k:sum(v.values()) for k,v in amb_segm_test.items()}
    amb = sum(amb_segm_test_freq.values())
    corr_amb = 0 # number of correct ambigous

    not_amb_segm_test = {k:v for k,v in test_dict.items() if len(v)==1}
    seen_freq = {k:v.values()[0] for k,v in not_amb_segm_test.items() if k in train_lexicon_w.keys()}
    seen = sum(seen_freq.values())
    corr_seen = 0 # number of correct seen words

    unseen_freq = {k:v.values()[0] for k,v in not_amb_segm_test.items() if not k in train_lexicon_w.keys()}
    unseen = sum(unseen_freq.values())

    unseen_m_freq = {k:v.values()[0] for k,v in not_amb_segm_test.items() if ( not k in train_lexicon_w.keys() and not all(m in train_lexicon_m.keys() for m in v.keys()[0].split(' ')) )}
    unseen_m = sum(unseen_m_freq.values())
    corr_unseen_m = 0 # number of correct unseen words - new morphs

    unseen_new_comb = unseen - unseen_m
    corr_unseen_comb = 0 # number of correct unseen words - new combinations
    
    gold_f.seek(0)
    for i,line in enumerate(gold_f):
        #if i < 5:
        if len(line.strip()) !=0:
            try:
                line = line.strip().lower() if lowercase else line.strip()
                lineitems = line.split('\t')
                w = lineitems[input]
                w_segm = lineitems[pred]
                lines = str(i)
                w_segm_morfs = w_segm.split(' ')
            
            except:
                print i, line
            
#                # remove diacritic
#                if unicodedata.combining(w[0]):
#                    w = w[1:]

            allc += 1
            if pred_dict_ext[(w,allc)] == w_segm:
                corr += 1
                
                if w in amb_segm_test_freq.keys():
                    corr_amb += 1
                else:
                    if w not in train_lexicon_w.keys():
                        
                        if all(m in train_lexicon_m.keys() for m in w_segm_morfs):
                            corr_unseen_comb += 1
                        else:
                            corr_unseen_m += 1
                    else:
                        corr_seen += 1
                    
                        
            else:

                if (w,pred_dict_ext[(w,allc)], w_segm) not in errors.keys():
                    errors[(w,pred_dict_ext[(w,allc)], w_segm)] = [lines]
                else:
                    errors[(w,pred_dict_ext[(w,allc)], w_segm)].append(lines)
                    
                    
    with codecs.open(file_out,'w','utf-8') as f:
        
        # Print statistics
        
        f.write("\nDATA:")
        f.write("\n   TRAIN  TEST:")
        f.write("\n# of target segment tokens %d  %d" % (sum(train_lexicon_m.values()), sum(test_lexicon_m.values())))
        f.write("\n# of source word tokens %d  %d" % (sum(train_lexicon_w.values()), sum(test_lexicon_w.values())))
        f.write("\n# of ambigous source word tokens        %d (%.2f%%)" % (amb, float(amb)/float(sum(test_lexicon_w.values()))*100))
        f.write("\n# of seen source word tokens        %d (%.2f%%)" % (seen, float(seen)/float(sum(test_lexicon_w.values()))*100))
        f.write("\n# of unseen source word tokens      %d (%.2f%%)" % (unseen, float(unseen)/float(sum(test_lexicon_w.values()))*100))
        f.write("\n# of unseen source word tokens - new target segments       %d (%.2f%%)" % (unseen_m, float(unseen_m)/float(sum(test_lexicon_w.values()))*100))
        f.write("\n# of unseen word tokens - new combination        %d (%.2f%%)" % (unseen_new_comb, float(unseen_new_comb)/float(sum(test_lexicon_w.values()))*100))


        f.write("============================================================================================")
    
        f.write("\nPERFORMANCE:")
        f.write("\n        Number of predictions total: %d" % allc)
        f.write("\nNumber of correct predictions total: %d (%.2f%%)" % (corr, float(corr)/float(allc)*100))
        f.write("\n                         - ambigous: %d (%.2f%%)"
                % (corr_amb, float(corr_amb)/float(amb)*100))
        if seen !=0:
            f.write("\n                   - seen words: %d (%.2f%%)"
                  % (corr_seen, float(corr_seen)/float(seen)*100))
        f.write("\n            - unseen (new morphemes): %d (%.2f%%)"
                % (corr_unseen_m, float(corr_unseen_m)/float(unseen_m)*100))
        f.write("\n            - unseen (new combination): %d (%.2f%%)"
                % (corr_unseen_comb, float(corr_unseen_comb)/float(unseen_new_comb)*100))
    
        f.write("============================================================================================")

    with codecs.open(file_out_errors,'w','utf-8') as f:
        #f.write("\n\nERRORS:\n")
        f.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'
                .format("word","prediction", "gold", "err_freq", "ambigous?", "word_seen?", "new_morphemes?", "lines(test)"))
        orderd_w = sorted(errors.keys(), key=lambda v: v[1], reverse=True)
        for (w,pred,true_pred) in orderd_w:
            seen_w = w in train_lexicon_w.keys()
            seen_w, new_m = 'NA','NA'
            amb_type = w in amb_segm_test.keys()
            if not amb_type:
                seen_w = w in train_lexicon_w.keys()
                if not seen_w:
                    new_m = not all(m in train_lexicon_m.keys() for m in true_pred.split(' '))
            f.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'
                    .format(w, pred, true_pred, len(errors[(w,pred,true_pred)]), amb_type, seen_w, new_m, ", ".join(errors[(w,pred,true_pred)])))

if __name__ == "__main__":
    arguments = docopt(__doc__)
    print arguments
    
    trainin = arguments['TRAIN_DATA']
    gold = arguments['TEST_DATA']
    predict = arguments['PREDICTIONS']
    file_out = arguments['RESULTS_FILE']
    file_out_errors = arguments['RESULTS_ERRORS_FILE']
    input_format_arg = arguments['--input_format']
    input_format=[int(col) for col in input_format_arg.split(',')]
    
    if arguments['eval']:
        evaluate(trainin,gold,predict,file_out,file_out_errors, input_format,arguments['--lowercase'],arguments['--extended_train_data'])
    elif arguments['eval_baseline']:
        evaluate_baseline(trainin,gold, input_format,arguments['--lowercase'],arguments['--pred_file'])
    elif arguments['eval_ambiguity']:
        evaluate_ambiguity(trainin,gold,predict,file_out,file_out_errors, input_format,arguments['--lowercase'])
    else:
        print "Unknown option"


