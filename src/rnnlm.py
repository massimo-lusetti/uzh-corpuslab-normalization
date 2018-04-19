#!/usr/bin/env python
# -*- coding: utf-8 -*
"""Trains LSTM language model.

Usage:
  rnnlm.py train [--dynet-seed SEED] [--dynet-mem MEM]
  [--input=INPUT] [--hidden=HIDDEN] [--feat-input=FEAT] [--layers=LAYERS] [--segments]
  [--dropout=DROPOUT] [--epochs=EPOCHS] [--patience=PATIENCE] [--optimization=OPTIMIZATION]
  MODEL_FOLDER --train_path=TRAIN_FILE --dev_path=DEV_FILE
  rnnlm.py test [--dynet-seed SEED] [--dynet-mem MEM]
  [--input=INPUT] [--hidden=HIDDEN] [--feat-input=FEAT] [--layers=LAYERS] [--segments]
  MODEL_FOLDER --test_path=TEST_FILE
  
Arguments:
  MODEL_FOLDER  save/read model folder where also eval results are written to, possibly relative to RESULTS_FOLDER

Options:
  -h --help                     show this help message and exit
  --dynet-seed SEED             DyNET seed
  --dynet-mem MEM               allocates MEM bytes for DyNET [default: 500]
  --input=INPUT                 input vector dimensions [default: 100]
  --hidden=HIDDEN               hidden layer dimensions [default: 200]
  --feat-input=FEAT             feature input vector dimension [default: 20]
  --layers=LAYERS               amount of layers in LSTMs  [default: 1]
  --dropout=DROPOUT             amount of dropout in LSTMs [default: 0]
  --patience=PATIENCE           patience for early stopping [default: 10]
  --epochs=EPOCHS               number of training epochs   [default: 10]
  --optimization=OPTIMIZATION   chosen optimization method ADAM/SGD/ADAGRAD/MOMENTUM/ADADELTA [default: SGD]
  --train_path=TRAIN_FILE       train set path, possibly relative to DATA_FOLDER, only for training
  --dev_path=DEV_FILE           dev set path, possibly relative to DATA_FOLDER, only for training
  --test_path=TEST_FILE         test set path, possibly relative to DATA_FOLDER, only for evaluation
  --segments                    run LM over segments instead of chars
"""

from __future__ import division
from docopt import docopt
import os
import sys
import codecs
import random
import progressbar
import time
from collections import Counter, defaultdict


import dynet as dy
import numpy as np
import os
from itertools import izip


# Default paths
SRC_FOLDER = os.path.dirname(__file__)
RESULTS_FOLDER = os.path.join(SRC_FOLDER, '../results')
DATA_FOLDER = os.path.join(SRC_FOLDER, '../data/')



# Model defaults
BOUNDARY_CHAR = u' '
BEGIN_CHAR   = u'<s>'
STOP_CHAR   = u'</s>'
UNK_CHAR = '<unk>'
MAX_PRED_SEQ_LEN = 50
OPTIMIZERS = {'ADAM'    : lambda m: dy.AdamTrainer(m, lam=0.0, alpha=0.0001,
                                                   beta_1=0.9, beta_2=0.999, eps=1e-8),
            'SGD'     : dy.SimpleSGDTrainer,
            'ADADELTA': dy.AdadeltaTrainer}


### IO handling and evaluation

def check_path(path, arg_name, is_data_path=True):
    if not os.path.exists(path):
        prefix = DATA_FOLDER if is_data_path else RESULTS_FOLDER
        tmp = os.path.join(prefix, path)
        if os.path.exists(tmp):
            path = tmp
        else:
            if is_data_path:
                print '%s incorrect: %s and %s' % (arg_name, path, tmp)
                raise ValueError
            else: #results path
                os.makedirs(tmp)
                path = tmp
    return path



def log_to_file(log_file_name, e, train_perplexity, dev_perplexity):
    # if first write, add headers
    if e == 0:
        log_to_file(log_file_name, 'epoch', 'train_perplexity', 'dev_perplexity')
    
    with open(log_file_name, "a") as logfile:
        logfile.write("{}\t{}\t{}\n".format(e, train_perplexity, dev_perplexity))


# represents a bidirectional mapping from strings to ints
class Vocab(object):
    def __init__(self, w2i):
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.iteritems()}
    
    def save(self, vocab_path):
        with codecs.open(vocab_path, 'w', 'utf-8') as fh:
            for w,i in self.w2i.iteritems():
                fh.write(u'{}\t{}\n'.format(w,i))
        return
    
    @classmethod
    def from_list(cls, words):
        w2i = {}
        idx = 0
        for word in words:
            w2i[word] = idx
            idx += 1
        return Vocab(w2i)
    
    @classmethod
    def from_file(cls, vocab_fname):
        w2i = {}
        with codecs.open(vocab_fname, 'r', 'utf-8') as fh:
            for line in fh:
                word, idx = line.strip().split()
                w2i[word] = int(idx)
        return Vocab(w2i)

    def size(self): return len(self.w2i.keys())

def read(filename, over_segs=False):
    """
        Read a file where each line is of the form "word1 word2 ..."
        Yields lists of the lines from file
        """
    with codecs.open(filename, encoding='utf8') as fh:
        for line in fh:
            splt = line.strip().split('\t')
            assert len(splt) == 2, 'bad line: ' + line.encode('utf8') + '\n'
            input, output = splt
            # language model is trained on the target side of the corpus
            if over_segs:
                # Segments
                yield output.split(BOUNDARY_CHAR)
            else:
                # Chars
                yield [c for c in output]

def write_results_file(hyper_params, perplexity, train_path, test_path, output_file_path):
    
    # write hyperparams, micro + macro avg. accuracy
    with codecs.open(output_file_path, 'w', encoding='utf8') as f:
        f.write('train/model path = ' + str(train_path) + '\n')
        f.write('test path = ' + str(test_path) + '\n')
        
        for param in hyper_params:
            f.write(param + ' = ' + str(hyper_params[param]) + '\n')
        
        f.write('Perplexity = ' + str(perplexity) + '\n')

    print 'wrote results to: ' + output_file_path + '\n'
    return



class RNNLanguageModel(object):
    def __init__(self, model, model_hyperparams, train_data=None):
        
        self.hyperparams = model_hyperparams
        
        if train_data:
            # Build vocabulary from the train data
            self.build_vocabulary(train_data)
        else:
            # Load vocabulary of pretrained model
            if os.path.exists(self.hyperparams['VOCAB_PATH']):
                self.vocab = Vocab.from_file(self.hyperparams['VOCAB_PATH'])
            else:
                print 'No vocabulary path'
                raise ValueError

        #self.BEGIN   = self.vocab.w2i[BEGIN_CHAR]
        #self.STOP   = self.vocab.w2i[STOP_CHAR]
        self.UNK       = self.vocab.w2i[UNK_CHAR]
        self.hyperparams['VOCAB_SIZE'] = self.vocab.size()
        
        self.build_model(model)
            
        print 'Model Hypoparameters:'
        for k, v in self.hyperparams.items():
            print '{:20} = {}'.format(k, v)
        print
        
    def build_vocabulary(self, train_data):
        
        # Build vocabulary over items - chars or segments
        items = list(set([c for w in train_data for c in w])) + [STOP_CHAR] + [UNK_CHAR] + [BEGIN_CHAR]
        print 'Example of vocabulary items:' + u', '.join(items[:10])
        self.vocab = Vocab.from_list(items)
    
    def build_model(self, model):
        
        # LSTM
        self.RNN  = dy.CoupledLSTMBuilder(self.hyperparams['LAYERS'], self.hyperparams['INPUT_DIM'], self.hyperparams['HIDDEN_DIM'], model)
        
        # embedding lookups for vocabulary
        self.VOCAB_LOOKUP  = model.add_lookup_parameters((self.hyperparams['VOCAB_SIZE'], self.hyperparams['INPUT_DIM']))

        # softmax parameters
        self.R = model.add_parameters((self.hyperparams['VOCAB_SIZE'], self.hyperparams['HIDDEN_DIM']))
        self.bias = model.add_parameters(self.hyperparams['VOCAB_SIZE'])
        
        
        print 'Model dimensions:'
        print ' * VOCABULARY EMBEDDING LAYER: IN-DIM: {}, OUT-DIM: {}'.format(self.hyperparams['VOCAB_SIZE'], self.hyperparams['INPUT_DIM'])
        print
        print ' * LSTM: IN-DIM: {}, OUT-DIM: {}'.format(self.hyperparams['INPUT_DIM'], self.hyperparams['HIDDEN_DIM'])
        print ' LSTM has {} layer(s)'.format(self.hyperparams['LAYERS'])
        print
        print ' * SOFTMAX: IN-DIM: {}, OUT-DIM: {}'.format(self.hyperparams['HIDDEN_DIM'], self.hyperparams['VOCAB_SIZE'])
        print
    
    def BuildLMGraph(self, input):
    
        R = dy.parameter(self.R)   # hidden to vocabulary
        bias = dy.parameter(self.bias)
        s = self.RNN.initial_state()
        
        input = [BEGIN_CHAR] + input + [STOP_CHAR]
        inputs_id = [self.vocab.w2i.get(c, self.UNK) for c in input]
        inputs_emb = [self.VOCAB_LOOKUP[c_id] for c_id in inputs_id]

        inputs = inputs_emb[:-1]
        true_preds = inputs_id[1:]
        
        states = s.transduce(inputs)
        prob_ts = (bias + (R * s_t) for s_t in states)
        losses = [dy.pickneglogsoftmax(prob_t,true_pred) for prob_t, true_pred in izip(prob_ts, true_preds)]

#        losses = []
#        s = s.add_input(self.VOCAB_LOOKUP[inputs_id[0]])
#        for wid in inputs_id[1:]:
#            scores = bias + (R * s.output())
#            loss = dy.pickneglogsoftmax(scores, wid)
#            losses.append(loss)
#            s = s.add_input(self.VOCAB_LOOKUP[wid])

        return dy.esum(losses)

if __name__ == "__main__":
    arguments = docopt(__doc__)
    print arguments
    
    np.random.seed(123)
    random.seed(123)

    model_folder = check_path(arguments['MODEL_FOLDER'], 'MODEL_FOLDER', is_data_path=False)

    if arguments['train']:
        
        print '=========TRAINING:========='

        assert (arguments['--train_path']!=None) & (arguments['--dev_path']!=None)
        
        # load data
        print 'Loading data...'
        over_segs = arguments['--segments']
        train_path = check_path(arguments['--train_path'], 'train_path')
        train_data = list(read(train_path, over_segs))
        print 'Train data has {} examples'.format(len(train_data))
        dev_path = check_path(arguments['--train_path'], 'train_path')
        dev_data = list(read(dev_path, over_segs))
        print 'Dev data has {} examples'.format(len(dev_data))
        
        print 'Checking if any special symbols in data...'
        for data, name in [(train_data, 'train'), (dev_data, 'dev')]:
            data = set([c for w in data for c in w])
            for c in [BEGIN_CHAR, STOP_CHAR, UNK_CHAR]:
                assert c not in data
            print '{} data does not contain special symbols'.format(name)
        print
    
        log_file_name   = model_folder + '/log.txt'
        best_model_path  = model_folder + '/bestmodel.txt'
        vocab_path = model_folder + '/vocab.txt'
        output_file_path = model_folder + '/best.dev'
        
        # Model hypoparameters
        model_hyperparams = {'INPUT_DIM': int(arguments['--input']),
                            'HIDDEN_DIM': int(arguments['--hidden']),
                            #'FEAT_INPUT_DIM': int(arguments['--feat-input']),
                            'LAYERS': int(arguments['--layers']),
                            'DROPOUT': float(arguments['--dropout']),
                            'VOCAB_PATH': vocab_path}
    
        print 'Building model...'
        model = dy.Model()
        lm = RNNLanguageModel(model, model_hyperparams, train_data)

        # Training hypoparameters
        train_hyperparams = {'MAX_PRED_SEQ_LEN': MAX_PRED_SEQ_LEN,
                            'OPTIMIZATION': arguments['--optimization'],
                            'EPOCHS': int(arguments['--epochs']),
                            'PATIENCE': int(arguments['--patience']),
                            'BEAM_WIDTH': 1}
        print 'Train Hypoparameters:'
        for k, v in train_hyperparams.items():
            print '{:20} = {}'.format(k, v)
        print

        trainer = OPTIMIZERS[train_hyperparams['OPTIMIZATION']]
        trainer = trainer(model)

        best_dev_perplexity = 999.
#        patience = 0

        # progress bar init
        widgets = [progressbar.Bar('>'), ' ', progressbar.ETA()]
        train_progress_bar = progressbar.ProgressBar(widgets=widgets, maxval=train_hyperparams['EPOCHS']).start()

        for epoch in xrange(train_hyperparams['EPOCHS']):
            print 'Start training...'
            then = time.time()

            # compute loss for each sample and update
            random.shuffle(train_data)
            train_loss = 0.
            loss_processed = 0 # for intermidiate reporting
            train_units_processed = 0 # for intermidiate reporting
            train_units = 0
            
            for i, input in enumerate(train_data, 1):
                # comp graph for each training example
                dy.renew_cg()
                loss = lm.BuildLMGraph(input)
                train_loss += loss.scalar_value()
                loss_processed += loss.scalar_value()
                loss.backward()
                trainer.update()
                train_units_processed += len(input) + 2
                train_units += len(input) + 2
                
                # intermediate report on perplexity
                if i % 20000 == 0:
                    trainer.status()
                    print 'processed: {}, loss: {:.3f}, perplexity: {:.3f}'.format(i, loss_processed/train_units_processed, np.exp(loss_processed/train_units_processed))
                    train_units_processed = 0
                    loss_processed = 0

            avg_train_loss = train_loss/len(train_data)
            train_perplexity = np.exp(train_loss/train_units)

            print '\t...finished in {:.3f} sec'.format(time.time() - then)

            # get dev accuracy
            print 'evaluating on dev...'
            then = time.time()

            dev_loss = 0.
            dev_units = 0
            for input in dev_data:
                loss = lm.BuildLMGraph(input)
                dev_loss += loss.scalar_value()
                dev_units += len(input) + 2
            
            avg_dev_loss = dev_loss/len(dev_data)
            dev_perplexity = np.exp(dev_loss/dev_units)
            
            print '\t...finished in {:.3f} sec'.format(time.time() - then)

            if dev_perplexity < best_dev_perplexity:
                best_dev_perplexity = dev_perplexity
                # save best model
                model.save(best_model_path)
                print 'saved new best model to {}'.format(best_model_path)
#                patience = 0
#            else:
#                patience += 1

            print ('epoch: {0} avg train loss: {1:.4f} avg dev loss: {2:.4f} train perplexity: {3:.4f} '
                   'dev perplexity: {4:.4f} best dev perplexity: {5:.4f} '
                   ).format(epoch, avg_train_loss, avg_dev_loss, train_perplexity, dev_perplexity, best_dev_perplexity)

            log_to_file(log_file_name, epoch, train_perplexity, dev_perplexity)

#            if patience == max_patience:
#                print 'out of patience after {} epochs'.format(epoch)
#                train_progress_bar.finish()
#                break
            # finished epoch
            train_progress_bar.update(epoch)
    
        print 'finished training.'
        # save vocab file
        lm.vocab.save(vocab_path)
        
        # save best dev model parameters
        write_results_file(lm.hyperparams, best_dev_perplexity, train_path, dev_path, output_file_path)

    elif arguments['test']:
        print '=========EVALUATION ONLY:========='
        # requires test path, model path of pretrained path and results path where to write the results to
        assert arguments['--test_path']!=None
        
        print 'Loading data...'
        over_segs = arguments['--segments']
        test_path = check_path(arguments['--test_path'], '--test_path')
        test_data = list(read(test_path, over_segs))
        print 'Test data has {} examples'.format(len(test_data))
        
        print 'Checking if any special symbols in data...'
        data = set([c for w in test_data for c in w])
        for c in [BEGIN_CHAR, STOP_CHAR, UNK_CHAR]:
            assert c not in data
        print 'Test data does not contain special symbols'

        best_model_path  = model_folder + '/bestmodel.txt'
        vocab_path = model_folder + '/vocab.txt'
        output_file_path = model_folder + '/best.test'

        model_hyperparams = {'INPUT_DIM': int(arguments['--input']),
                            'HIDDEN_DIM': int(arguments['--hidden']),
                            #'FEAT_INPUT_DIM': int(arguments['--feat-input']),
                            'LAYERS': int(arguments['--layers']),
                            'DROPOUT': float(arguments['--dropout']),
                            'VOCAB_PATH': vocab_path}
        
        model = dy.Model()
        lm = RNNLanguageModel(model, model_hyperparams)

        print 'trying to load model from: {}'.format(best_model_path)
        model.populate(best_model_path)

        dev_loss = 0.
        dev_units = 0
        for input in test_data:
            loss = lm.BuildLMGraph(input)
            dev_loss += loss.scalar_value()
            dev_units += len(input) + 2
        perplexity = np.exp(dev_loss/dev_units)
        print 'Perplexity: {}'.format(perplexity)

        write_results_file(lm.hyperparams, perplexity, best_model_path, test_path, output_file_path)
