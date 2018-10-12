#!/usr/bin/env python
# -*- coding: utf-8 -*
""" This file evaluates results with details over seen/unseen segments(morphemes/words)

Usage:
  accuracy-det.py [--input_format=INPUT_FORMAT] [--lowercase=LOW]
  TRAIN_DATA TEST_DATA PREDICTIONS RESULTS_FILE RESULTS_ERRORS_FILE

Arguments:
TRAIN_DATA          train file path
TEST_DATA           test file path
PREDICTIONS         path for the predictions for the test data
RESULTS_FILE        path to save the evaluation
RESULTS_ERRORS_FILE path to save errors

Options:
  -h --help                     show this help message and exit
  --input_format=INPUT_FORMAT   coma-separated list of input, output, features columns [default: 0,1]
  --lowercase=LOW               use lowercased data [default: True]
"""

__author__ = 'Tanya'
from docopt import docopt
import codecs
import unicodedata
from collections import defaultdict


def evaluate(trainin,gold,predict,file_out,file_out_errors,input_format,lowercase=False):
    train_lexicon_m = defaultdict(int)
    test_lexicon_m = defaultdict(int)
    train_lexicon_w = defaultdict(int)
    test_lexicon_w = defaultdict(int)
    test_dict = {}

    unseen = 0
    unseen_new_m = 0
    unseen_new_comb = 0

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
            if not word in train_lexicon_w.keys():
                unseen += 1
                if all(m in train_lexicon_m.keys() for m in morfs):
                    #if word == 'ghode': print 'Yes'
                    unseen_new_comb +=1
                #print line.encode('utf8')
                else:
                    unseen_new_m += 1
            if not word in test_dict.keys():
                test_dict[word] = {}
                test_dict[word][segm]=1
            else:
                if segm not in test_dict[word].keys():
                    test_dict[word][segm]=1
                else:
                    test_dict[word][segm]+=1



    # Collect predictions
    predict_f = codecs.open(predict,'r','utf-8')
    pred_dict = {}
    test_lines = {}
    pred_dict_ext = {}
    for j, line in enumerate(predict_f):
        line = line.strip().lower() if lowercase else line.strip()
        w, w_segm = line.split('\t')
        pred_dict_ext[(w,j+1)] = w_segm
        # from space separated chars to merged chars, special treatement to the boundary symbol
        if w not in pred_dict:
            
            pred_dict[w] = w_segm
            test_lines[w] = str(j+1)



    # Evaluation

    amb_segm_test = {k:v for k,v in test_dict.items() if len(v)>1}
    print amb_segm_test
    print len({k:sum(v.values()) for k,v in test_dict.items() if len(v)>1})
#    amb_segm_test_incorrect = {k:sum(freq) for k,v in test_dict.items() if all(pred_dict.get(k)!=seg for seg,freq in v.items()) and len(v)>1}
#    print len(amb_segm_test_incorrect)

    errors = {}
    test_corpus_lines = {}

        
    #LM Evaluation
    allc = 0 # total number of predictions  (that is the number of words in the input (gold))
    corr = 0 # total number of correct predictions
    corr_seen = 0 # number of correct seen words
    corr_unseen = 0 # number of correct unseen words
    corr_unseen_m = 0 # number of correct unseen words - new morphs
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

                if w not in train_lexicon_w.keys():
                    corr_unseen += 1
                    
                    if all(m in train_lexicon_m.keys() for m in w_segm_morfs):
                        corr_unseen_comb += 1
                    else:
                        corr_unseen_m += 1
                else:
                    corr_seen += 1
                        
            else:
#                print i, line, pred_dict[w]
                seen_w = w in train_lexicon_w.keys()
                #new_m = not all(m in train_lexicon_m.keys() for m in w_segm_morfs)
                amb_type = w in amb_segm_test.keys()
#                if amb_type == True:
#                    amb_type_1cor_pred = not w in amb_segm_test_incorrect.keys()
#                else:
#                    amb_type_1cor_pred = 'NA'

                if w not in errors.keys():
                    errors[w] = {}
                    errors[w][w_segm] = 1
                    test_corpus_lines[w] = {}
                    test_corpus_lines[w][w_segm] = [lines]
                else:
                    if w_segm not in errors[w].keys():
                        errors[w][w_segm] =  1
                        test_corpus_lines[w][w_segm] = [lines]
                    else:
                        errors[w][w_segm] +=  1
                        test_corpus_lines[w][w_segm].append(lines)

    with codecs.open(file_out,'w','utf-8') as f:
        
        # Print statistics
        seen = sum(test_lexicon_w.values()) - unseen
        
        f.write("\nDATA:")
        f.write("\n   TRAIN  TEST:")
        f.write("\n# of morpheme tokens %d  %d" % (sum(train_lexicon_m.itervalues()), sum(test_lexicon_m.itervalues())))
        f.write("\n# of word tokens %d  %d" % (sum(train_lexicon_w.itervalues()), sum(test_lexicon_w.itervalues())))
        f.write("\n# of seen word tokens        %d (%.2f%%)" % (seen, float(seen)/float(sum(test_lexicon_w.itervalues()))*100))
        f.write("\n# of unseen word tokens      %d (%.2f%%)" % (unseen, float(unseen)/float(sum(test_lexicon_w.itervalues()))*100))
        f.write("\n# of unseen word tokens - new morphemes       %d (%.2f%%)" % (unseen_new_m, float(unseen_new_m)/float(sum(test_lexicon_w.itervalues()))*100))
        f.write("\n# of unseen word tokens - new combination        %d (%.2f%%)" % (unseen_new_comb, float(unseen_new_comb)/float(sum(test_lexicon_w.itervalues()))*100))


        f.write("============================================================================================")
    
        f.write("\nPERFORMANCE:")
        f.write("\n        Number of predictions total: %d" % allc)
        f.write("\nNumber of correct predictions total: %d (%.2f%%)" % (corr, float(corr)/float(allc)*100))
        if seen !=0:
            f.write("\n                            - seen words: %d (%.2f%%)"
                  % (corr_seen, float(corr_seen)/float(seen)*100))
        f.write("\n                         - new morphemes: %d (%.2f%%)"
                % (corr_unseen_m, float(corr_unseen_m)/float(unseen_new_m)*100))
        f.write("\n                       - new combination: %d (%.2f%%)"
                % (corr_unseen_comb, float(corr_unseen_comb)/float(unseen_new_comb)*100))
    
        f.write("============================================================================================")

    with codecs.open(file_out_errors,'w','utf-8') as f:
        #f.write("\n\nERRORS:\n")
        f.write(u'{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\n'
                .format("ID","word","prediction", "gold", "err_freq", "word_seen?", "new_morphemes?", "ambigous_type?", "ambigous_type_one_correct_predict?", "lines(train)"))
        orderd_w = sorted(errors.keys(), key=lambda x: sum(errors[x][y] for y in errors[x]), reverse=True)
        for w in orderd_w:
            seen_w = w in train_lexicon_w.keys()
            #new_m = not all(m in train_lexicon_m.keys() for m in w_segm_morfs)
            amb_type = w in amb_segm_test.keys()
#            if amb_type == True:
#                amb_type_1cor_pred = not w in amb_segm_test_incorrect.keys()
#            else:
#                amb_type_1cor_pred = 'NA'
            for gold_segm in errors[w].keys():
                new_m = not all(m in train_lexicon_m.keys() for m in gold_segm.split(' '))
                f.write(u'{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n'
                                .format(test_lines[w], w, pred_dict[w], gold_segm, errors[w][gold_segm], seen_w, new_m, amb_type, ", ".join(test_corpus_lines[w][gold_segm])))

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

    evaluate(trainin,gold,predict,file_out,file_out_errors, input_format,arguments['--lowercase'])

