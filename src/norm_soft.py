#!/usr/bin/env python
# -*- coding: utf-8 -*
"""Trains encoder-decoder model with soft attention.

Usage:
  norm_soft.py train [--dynet-seed SEED] [--dynet-mem MEM]
    [--input=INPUT] [--hidden=HIDDEN] [--feat-input=FEAT] [--layers=LAYERS]
    [--dropout=DROPOUT] [--epochs=EPOCHS] [--patience=PATIENCE] [--optimization=OPTIMIZATION]
    MODEL_FOLDER --train_path=TRAIN_FILE --dev_path=DEV_FILE
  norm_soft.py test [--dynet-seed SEED] [--dynet-mem MEM]
    [--input=INPUT] [--hidden=HIDDEN] [--feat-input=FEAT] [--layers=LAYERS]
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
  --epochs=EPOCHS               number of training epochs   [default: 30]
  --patience=PATIENCE           patience for early stopping [default: 10]
  --optimization=OPTIMIZATION   chosen optimization method ADAM/SGD/ADAGRAD/MOMENTUM/ADADELTA [default: SGD]
  --train_path=TRAIN_FILE       train set path, possibly relative to DATA_FOLDER, only for training
  --dev_path=DEV_FILE           dev set path, possibly relative to DATA_FOLDER, only for training
  --test_path=TEST_FILE         test set path, possibly relative to DATA_FOLDER, only for evaluation
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


# Default paths
SRC_FOLDER = os.path.dirname(__file__)
RESULTS_FOLDER = os.path.join(SRC_FOLDER, '../results')
DATA_FOLDER = os.path.join(SRC_FOLDER, '../data/')



# Model defaults
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

def load_data(filename):
    """ Load data from file
        
        filename (str):   file containing input/output data, structure (tab-separated):
        input    output
        return tuple (output, input) where each element is a list
        where each element in the list is one example
        """
    
    print 'loading data from file:', filename
    
    inputs, outputs = [], []
    
    with codecs.open(filename, encoding='utf8') as f:
        for line in f:
            splt = line.strip().split('\t')
            assert len(splt) == 2, 'bad line: ' + line.encode('utf8') + '\n'
            input, output = splt #can be adapted to the task with features
            inputs.append(input)
            outputs.append(output)

    tup = (inputs, outputs)
    print 'found', len(outputs), 'examples'
    return tup

def get_accuracy_predictions(ti, test_data):
    correct = 0.
    final_results = []
    for input,output in test_data.iter():
        loss, prediction = ti.transduce(input)
        if prediction == output:
            correct += 1
        final_results.append((input,prediction))  # pred expected as list
    accuracy = correct / test_data.length
    return accuracy, final_results

def write_results_file(hyper_params, accuracy, train_path, test_path, output_file_path, final_results):
    
    # write hyperparams, micro + macro avg. accuracy
    with codecs.open(output_file_path, 'w', encoding='utf8') as f:
        f.write('train/model path = ' + str(train_path) + '\n')
        f.write('test path = ' + str(test_path) + '\n')
        
        for param in hyper_params:
            f.write(param + ' = ' + str(hyper_params[param]) + '\n')
        
        f.write('Prediction Accuracy = ' + str(accuracy) + '\n')
    
    
    predictions_path = output_file_path + '.predictions'
    
    print 'len of predictions is {}'.format(len(final_results))
    with codecs.open(predictions_path, 'w', encoding='utf8') as predictions:
        for input, prediction in final_results:
            predictions.write(u'{0}\t{1}\n'.format(input, prediction))

    print 'wrote results to: ' + output_file_path + '\n' + output_file_path + '.evaluation' + '\n' + predictions_path
    return

def log_to_file(log_file_name, e, avg_train_loss, train_accuracy, dev_accuracy):
    # if first write, add headers
    if e == 0:
        log_to_file(log_file_name, 'epoch', 'avg_train_loss', 'train_accuracy', 'dev_accuracy')
    
    with open(log_file_name, "a") as logfile:
        logfile.write("{}\t{}\t{}\t{}\n".format(e, avg_train_loss, train_accuracy, dev_accuracy))


# represents a bidirectional mapping from strings to ints
class Vocab(object):
    def __init__(self, w2i):
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.iteritems()}
    
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
        with file(vocab_fname) as fh:
            for line in fh:
                word, idx = line.strip().split()
                w2i[word] = idx
        return Vocab(w2i)

    def size(self): return len(self.w2i.keys())

# class to handle data
class SoftDataSet(object):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.dataset = self.inputs, self.outputs
        self.length = len(self.outputs)
    
    def iter(self, indices=None, shuffle=False):
        zipped = zip(*self.dataset)
        if indices or shuffle:
            if not indices:
                indices = range(self.length)
            elif isinstance(indices, int):
                indices = range(indices)
            else:
                assert isinstance(indices, (list, tuple))
            if shuffle:
                random.shuffle(indices)
            zipped = [zipped[i] for i in indices]
        return zipped
    
    @classmethod
    def from_file(cls, path, *args, **kwargs):
        # returns a `SoftDataSet` with fields: inputs, outputs
        inputs, outputs = load_data(path)
        return cls(inputs, outputs, *args, **kwargs)

class SoftAttention(object):
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

        self.BEGIN   = self.vocab.w2i[BEGIN_CHAR]
        self.STOP   = self.vocab.w2i[STOP_CHAR]
        self.UNK       = self.vocab.w2i[UNK_CHAR]
        self.hyperparams['VOCAB_SIZE'] = self.vocab.size()
        
        self.build_model(model)
        
        print 'Model Hypoparameters:'
        for k, v in self.hyperparams.items():
            print '{:20} = {}'.format(k, v)
        print
        
    def build_vocabulary(self, train_data):
        
        chars = list(set([c for w in train_data.inputs for c in w] + [c for w in train_data.outputs for c in w])) + [STOP_CHAR] + [UNK_CHAR] + [BEGIN_CHAR]
        print 'Example of vocabulary items:' + u', '.join(chars[:10])
        self.vocab = Vocab.from_list(chars)
    

    def build_model(self, model):
        
        # BiLSTM for input
        self.fbuffRNN  = dy.CoupledLSTMBuilder(self.hyperparams['LAYERS'], self.hyperparams['INPUT_DIM'], self.hyperparams['HIDDEN_DIM'], model)
        self.bbuffRNN  = dy.CoupledLSTMBuilder(self.hyperparams['LAYERS'], self.hyperparams['INPUT_DIM'], self.hyperparams['HIDDEN_DIM'], model)
        
        # embedding lookups for vocabulary
        self.VOCAB_LOOKUP  = model.add_lookup_parameters((self.hyperparams['VOCAB_SIZE'], self.hyperparams['INPUT_DIM']))

        # decoder LSTM
        self.decoder = dy.CoupledLSTMBuilder(self.hyperparams['LAYERS'], self.hyperparams['INPUT_DIM'], self.hyperparams['HIDDEN_DIM'], model)

        # softmax parameters
        self.R = model.add_parameters((self.hyperparams['VOCAB_SIZE'], 3 * self.hyperparams['HIDDEN_DIM']))
        self.bias = model.add_parameters(self.hyperparams['VOCAB_SIZE'])
        
        # attention MLPs - Loung-style with extra v_a from Bahdanau
        
        # concatenation layer for h (hidden dim), c (2 * hidden_dim)
        self.W_c = model.add_parameters((3 * self.hyperparams['HIDDEN_DIM'], 3 * self.hyperparams['HIDDEN_DIM']))
        
        # attention MLP's - Bahdanau-style
        # concatenation layer for h_input (2*hidden_dim), h_output (hidden_dim)
        self.W__a = model.add_parameters((self.hyperparams['HIDDEN_DIM'], self.hyperparams['HIDDEN_DIM']))
        
        # concatenation layer for h (hidden dim), c (2 * hidden_dim)
        self.U__a = model.add_parameters((self.hyperparams['HIDDEN_DIM'], 2 * self.hyperparams['HIDDEN_DIM']))
        
        # concatenation layer for h_input (2*hidden_dim), h_output (hidden_dim)
        self.v__a = model.add_parameters((1, self.hyperparams['HIDDEN_DIM']))
        
        
        
        print 'Model dimensions:'
        print ' * VOCABULARY EMBEDDING LAYER: IN-DIM: {}, OUT-DIM: {}'.format(self.hyperparams['VOCAB_SIZE'], self.hyperparams['INPUT_DIM'])
        print
        print ' * ENCODER biLSTM: IN-DIM: {}, OUT-DIM: {}'.format(2*self.hyperparams['INPUT_DIM'], 2*self.hyperparams['HIDDEN_DIM'])
        print ' * DECODER LSTM: IN-DIM: {}, OUT-DIM: {}'.format(self.hyperparams['INPUT_DIM'], self.hyperparams['HIDDEN_DIM'])
        print ' All LSTMs have {} layer(s)'.format(self.hyperparams['LAYERS'])
        print
        print ' * SOFTMAX: IN-DIM: {}, OUT-DIM: {}'.format(self.hyperparams['HIDDEN_DIM'], self.hyperparams['VOCAB_SIZE'])
        print
    

    def bilstm_transduce(self, encoder_frnn, encoder_rrnn, input_char_vecs):
        
        # BiLSTM forward pass
        s_0 = encoder_frnn.initial_state()
        s = s_0
        frnn_outputs = []
        for c in input_char_vecs:
            s = s.add_input(c)
            frnn_outputs.append(s.output())
        
        # BiLSTM backward pass
        s_0 = encoder_rrnn.initial_state()
        s = s_0
        rrnn_outputs = []
        for c in reversed(input_char_vecs):
            s = s.add_input(c)
            rrnn_outputs.append(s.output())
        
        # BiLTSM outputs
        blstm_outputs = []
        for i in xrange(len(input_char_vecs)):
            blstm_outputs.append(dy.concatenate([frnn_outputs[i], rrnn_outputs[len(input_char_vecs) - i - 1]]))
        
        return blstm_outputs

    def transduce(self, input, _true_output=None, feats=None):
        
        # convert _true_output string to list of vocabulary indeces
        if _true_output:
            true_output = [self.vocab.w2i[a] for a in _true_output]
            true_output += [self.STOP]
            true_output = list(reversed(true_output))
        
        R = dy.parameter(self.R)   # hidden to vocabulary
        bias = dy.parameter(self.bias)
        W_c = dy.parameter(self.W_c)
        W__a = dy.parameter(self.W__a)
        U__a = dy.parameter(self.U__a)
        v__a = dy.parameter(self.v__a)
        
        
        # biLSTM encoder of input string
        input = [BEGIN_CHAR] + [c for c in input] + [STOP_CHAR]

        input_emb = []
        for char_ in reversed(input):
            char_id = self.vocab.w2i.get(char_, self.UNK)
            char_embedding = self.VOCAB_LOOKUP[char_id]
            input_emb.append(char_embedding)
        biencoder = self.bilstm_transduce(self.fbuffRNN, self.bbuffRNN, input_emb)
        
        losses = []
        output = []
        pred_history = [self.BEGIN] # <
        s = self.decoder.initial_state()
#        s=s0
        
        while not len(pred_history) == MAX_PRED_SEQ_LEN:
            # compute probability over vocabulary and choose a prediction
            # either from the true prediction at train time or based on the model at test time
            
            
            # decoder next state
            prev_pred_id = pred_history[-1]
            s = s.add_input(self.VOCAB_LOOKUP[prev_pred_id])
            
            # soft attention vector
            scores = [v__a * dy.tanh(W__a * s.output() + U__a * h_input) for h_input in biencoder]
            alphas = dy.softmax(dy.concatenate(scores))
            c = dy.esum([h_input * dy.pick(alphas, j) for j, h_input in enumerate(biencoder)])
            
            # softmax over vocabulary
            h_output = dy.tanh(W_c * dy.concatenate([s.output(), c]))
            probs = dy.softmax(R * h_output + bias)

            if _true_output is None:
                pred_id = np.argmax(probs.npvalue())
            else:
                pred_id = true_output.pop()
                
            losses.append(-dy.log(dy.pick(probs, pred_id)))
            pred_history.append(pred_id)
            
            if pred_id == self.STOP:
                break
            else:
                pred_char = self.vocab.i2w.get(pred_id,UNK_CHAR)
                output.append(pred_char)

        output = u''.join(output)
        return ((dy.average(losses) if losses else None), output)

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
        data_set = SoftDataSet
        train_path = check_path(arguments['--train_path'], 'train_path')
        train_data = data_set.from_file(train_path)
        print 'Train data has {} examples'.format(len(train_data))
        dev_path = check_path(arguments['-dev_path'], 'dev_path')
        dev_data = data_set.from_file(dev_path)
        print 'Dev data has {} examples'.format(len(dev_data))
    
        print 'Checking if any special symbols in data...'
        for data, name in [(train_data, 'train'), (dev_data, 'dev')]:
            data = set(data.inputs + data.outputs)
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
        ti = SoftAttention(model, model_hyperparams, train_data)

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

        best_dev_accuracy = -1.
        sanity_set_size = 100 # for speed - check prediction accuracy on train set
        patience = 0

        # progress bar init
        widgets = [progressbar.Bar('>'), ' ', progressbar.ETA()]
        train_progress_bar = progressbar.ProgressBar(widgets=widgets, maxval=train_hyperparams['EPOCHS']).start()
        
        for epoch in xrange(train_hyperparams['EPOCHS']):
            print 'Start training...'
            then = time.time()

            # compute loss for each sample and update
            train_loss = 0.  # total train loss
            avg_train_loss = 0.  # avg training loss

            for i, (input, output) in enumerate(train_data.iter(shuffle=True)):
                # here we do training
                dy.renew_cg()
                loss, _ = ti.transduce(input, output)
                if loss is not None:
                    train_loss += loss.scalar_value()
                    loss.backward()
                    trainer.update()

            avg_train_loss = train_loss / train_data.length

            print '\t...finished in {:.3f} sec'.format(time.time() - then)

            # get train accuracy
            print 'evaluating on train...'
            then = time.time()
            train_correct = 0.
            for i, (input, output) in enumerate(train_data.iter(indices=sanity_set_size)):
                _, prediction = ti.transduce(input)
                if prediction == output:
                    train_correct += 1
            train_accuracy = train_correct / sanity_set_size
            print '\t...finished in {:.3f} sec'.format(time.time() - then)
            
            # condition for displaying stuff like true outputs and predictions
            check_condition = (epoch > 0 and (epoch % 5 == 0 or epoch == epochs - 1))

            # get dev accuracy
            print 'evaluating on dev...'
            then = time.time()
            dev_correct = 0.
            dev_loss = 0.
            for input, output in dev_data.iter():
                loss, prediction = ti.transduce(input)
                if prediction == output:
                    dev_correct += 1
                    tick = 'V'
                else:
                    tick = 'X'
                if check_condition:
                    print 'TRUE:    ', output
                    print 'PRED:    ', prediction
                    print tick
                    print
                dev_loss += loss.scalar_value()
            dev_accuracy = dev_correct / dev_data.length
            avg_dev_loss = dev_loss / dev_data.length
            print '\t...finished in {:.3f} sec'.format(time.time() - then)

            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                # save best model
                model.save(best_model_path)
                print 'saved new best model to {}'.format(best_model_path)
                patience = 0
            else:
                patience += 1

            # found "perfect" model
            if dev_accuracy == 1:
                train_progress_bar.finish()
                break

            print ('epoch: {0} train loss: {1:.4f} dev loss: {2:.4f} dev accuracy: {3:.4f} '
                   'train accuracy: {4:.4f} best dev accuracy: {5:.4f} patience = {6}').format(epoch, avg_train_loss, avg_dev_loss, dev_accuracy, train_accuracy, best_dev_accuracy, patience)

            log_to_file(log_file_name, epoch, avg_train_loss, train_accuracy, dev_accuracy)

            if patience == max_patience:
                print 'out of patience after {} epochs'.format(epoch)
                train_progress_bar.finish()
                break
            # finished epoch
            train_progress_bar.update(epoch)
                
        print 'finished training.'
        # save vocab file
        lm.vocab.save(vocab_path)
    
        # save best dev model parameters and predictions
        accuracy, dev_results = get_accuracy_predictions(ti, dev_data)
        write_results_file(lm.hyperparams, accuracy, train_path, dev_path, output_file_path, dev_results)

    elif arguments['test']:
        print '=========EVALUATION ONLY:========='
        # requires test path, model path of pretrained path and results path where to write the results to
        assert arguments['--test_path']!=None

        print 'Loading data...'
        test_path = check_path(arguments['--test_path'], '--test_path')
        data_set = SoftDataSet
        test_data = data_set.from_file(test_path)
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
        ti = SoftAttention(model, model_hyperparams)

        print 'trying to load model from: {}'.format(best_model_path)
        model.populate(best_model_path)

        accuracy, test_results = get_accuracy_predictions(ti, test_data)
        print 'accuracy: {}'.format(accuracy)

        write_results_file(hyperparams, accuracy, train_path, test_path, output_file_path, test_results)
