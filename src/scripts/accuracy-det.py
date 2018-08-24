__author__ = 'Tanya'
# This file evaluates results of segmentation with details over seen/unseen morphemes

import re
import codecs
import sys
import unicodedata
from collections import defaultdict

# in a form w, w_segm, line_number
gold = codecs.open(sys.argv[1], "r", "utf-8") #$DATA/test.txt
# in a form w, w_segm, line_number
trainin = codecs.open(sys.argv[2], "r", "utf-8") #$DATA/train.txt
# prediction in a form w_segm
predict = codecs.open(sys.argv[3], "r", "utf-8") #input prediction
# prediction in a form w, w_segm
# sys.argv[4] - file to print errors

train_lexicon_m = defaultdict(int)
test_lexicon_m = defaultdict(int)
train_lexicon_w = defaultdict(int)
test_lexicon_w = defaultdict(int)
#train_dict = {}
test_dict = {}

unseen = 0
unseen_new_m = 0
unseen_new_comb = 0


# Read lexicon of training set
for i,line in enumerate(trainin):
    #if i < 40:
    line = re.sub(r'\n', '', line)
    if len(line) != 0:
        lineitems = line.lower().split('\t')
        word = lineitems[0]
        segm = lineitems[2]
        train_lexicon_w[word] += 1
#        morfs = lineitems[1].split('|')
##### Normalization adaption
        morfs = lineitems[2].split(' ')
        for m in morfs:
            train_lexicon_m[m] += 1

# Read lexicon of test set
for i,line in enumerate(gold):
    #if i < 40:
    line = re.sub(r'\n', '', line)
    if len(line) != 0:
        lineitems = line.lower().split('\t')
        word = lineitems[0]
        segm = lineitems[2]
        test_lexicon_w[word] += 1
        morfs = lineitems[2].split(' ')
#        morfs = lineitems[1].split('|')
##### Normalization adaption
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



# Print statistics
seen = sum(test_lexicon_w.values()) - unseen

print "\nDATA:"
print "\n   TRAIN  TEST:"
print "# of morpheme tokens %d  %d" % (sum(train_lexicon_m.itervalues()), sum(test_lexicon_m.itervalues()))
print "# of word tokens %d  %d" % (sum(train_lexicon_w.itervalues()), sum(test_lexicon_w.itervalues()))
print "# of seen word tokens        %d (%.2f%%)" % (seen, float(seen)/float(sum(test_lexicon_w.itervalues()))*100)
print "# of unseen word tokens      %d (%.2f%%)" % (unseen, float(unseen)/float(sum(test_lexicon_w.itervalues()))*100)
print "# of unseen word tokens - new morphemes       %d (%.2f%%)" % (unseen_new_m, float(unseen_new_m)/float(sum(test_lexicon_w.itervalues()))*100)
print "# of unseen word tokens - new combination        %d (%.2f%%)" % (unseen_new_comb, float(unseen_new_comb)/float(sum(test_lexicon_w.itervalues()))*100)



#line = re.sub(r'\n', '', line)

# Collect predictions
pred_dict = {}
test_lines = {}
for j, line in enumerate(predict):
    w, w_segm = line.strip().split('\t')
    # from space separated chars to merged chars, special treatement to the boundary symbol
    if w not in pred_dict:
        pred_dict[w] = w_segm
        test_lines[w] = str(j+1)
#### Nomralization adaptation
#    if w not in pred_dict:
#        pred_dict[w.replace(" ", "").strip()] = w_segm.replace(" ", "").strip()
#        test_lines[w.replace(" ", "").strip()] = j+1



# Evaluation

amb_segm_test = {k:v for k,v in test_dict.items() if len(v)>1}
print len(amb_segm_test)
amb_segm_test_incorrect = {k:v for k,v in test_dict.items()
    if all(pred_dict.get(k)!=seg for seg,freq in v.items()) and len(v)>1}
print len(amb_segm_test_incorrect)

errors = {}
test_corpus_lines = {}


with codecs.open(sys.argv[4],'w','utf-8') as f:
    
    #LM Evaluation
    allc = 0 # total number of predictions  (that is the number of words in the input (gold))
    corr = 0 # total number of correct predictions
    corr_seen = 0 # number of correct seen words
    corr_unseen = 0 # number of correct unseen words
    corr_unseen_m = 0 # number of correct unseen words - new morphs
    corr_unseen_comb = 0 # number of correct unseen words - new combinations
    
    gold.seek(0)
    for i,line in enumerate(gold):
        #if i < 5:
        line = re.sub(r'\n', '', line)
        if len(line) !=0:
#            w, w_segm, lines = line.split()
#             w_segm_morfs = w_segm.split('|')
#####Normalization adaption
            try:
                w, _, w_segm = line.lower().split('\t')
                lines = str(i)
                w_segm_morfs = w_segm.split(' ')
            
            except:
                print line
#####Normalization adaption
                    
            # remove diacritic
            if unicodedata.combining(w[0]):
                w = w[1:]
            
            allc += 1
            if pred_dict[w] == w_segm:
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
                seen_w = w in train_lexicon_w.keys()
                #new_m = not all(m in train_lexicon_m.keys() for m in w_segm_morfs)
                amb_type = w in amb_segm_test.keys()
                if amb_type == True:
                    amb_type_1cor_pred = not w in amb_segm_test_incorrect.keys()
                else:
                    amb_type_1cor_pred = 'NA'

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

with codecs.open(sys.argv[4],'w','utf-8') as f:
    #f.write("\n\nERRORS:\n")
    f.write(u'{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\n'
            .format("ID","word","prediction", "gold", "err_freq", "word_seen?", "new_morphemes?", "ambigous_type?", "ambigous_type_one_correct_predict?", "lines(train)"))
    orderd_w = sorted(errors.keys(), key=lambda x: sum(errors[x][y] for y in errors[x]), reverse=True)
    for w in orderd_w:
        seen_w = w in train_lexicon_w.keys()
        #new_m = not all(m in train_lexicon_m.keys() for m in w_segm_morfs)
        amb_type = w in amb_segm_test.keys()
        if amb_type == True:
            amb_type_1cor_pred = not w in amb_segm_test_incorrect.keys()
        else:
            amb_type_1cor_pred = 'NA'
        for gold_segm in errors[w].keys():
#            print test_lines[w]
#            print pred_dict[w]
#            print errors[w][gold_segm]
#            print ", ".join(test_corpus_lines[w][gold_segm])
#            new_m = not all(m in train_lexicon_m.keys() for m in gold_segm.split('|'))
#####Normalization adaption
            new_m = not all(m in train_lexicon_m.keys() for m in gold_segm.split(' '))
            f.write(u'{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\n'
                            .format(test_lines[w], w, pred_dict[w], gold_segm, errors[w][gold_segm], seen_w, new_m, amb_type, amb_type_1cor_pred, ", ".join(test_corpus_lines[w][gold_segm])))



print("===================================================================================================")


print("\nPERFORMANCE:")
print("\n        Number of predictions total: %d" % allc)
print("\nNumber of correct predictions total: %d (%.2f%%)" % (corr, float(corr)/float(allc)*100))
if seen !=0:
    print("\n                            - seen words: %d (%.2f%%)"
          % (corr_seen, float(corr_seen)/float(seen)*100))
print("\n                         - new morphemes: %d (%.2f%%)"
        % (corr_unseen_m, float(corr_unseen_m)/float(unseen_new_m)*100))
print("\n                       - new combination: %d (%.2f%%)"
        % (corr_unseen_comb, float(corr_unseen_comb)/float(unseen_new_comb)*100))

print("===================================================================================================")

