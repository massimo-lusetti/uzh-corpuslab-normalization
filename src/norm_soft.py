#!/usr/bin/env python
# -*- coding: utf-8 -*
"""Trains encoder-decoder model with soft attention.

Usage:
  norm_soft.py [--dynet-seed SEED] [--dynet-mem MEM]
  [--input=INPUT] [--hidden=HIDDEN] [--feat-input=FEAT] [--layers=LAYERS]
  [--dropout=DROPOUT] [--epochs=EPOCHS] [--patience=PATIENCE] [--optimization=OPTIMIZATION] [--eval]
  TRAIN_PATH DEV_PATH RESULTS_PATH [--test_path=TEST_PATH]

Arguments:
  TRAIN_PATH    destination path, possibly relative to "data/all/", e.g. task1/albanian-train-low
  DEV_PATH      development set path, possibly relative to "data/all/"
  RESULTS_PATH  results file to be written, possibly relative to "results"

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
  --optimization=OPTIMIZATION   chosen optimization method ADAM/SGD/ADAGRAD/MOMENTUM/ADADELTA [default: ADADELTA]
  --eval                        run evaluation without training
  --test_path=TEST_PATH         test set path
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
SRC_PATH = os.path.dirname(__file__)
RESULTS_PATH = os.path.join(SRC_PATH, '../results')
DATA_PATH = os.path.join(SRC_PATH, '../data/')



# Model defaults
BEGIN_CHAR   = u'≤'
STOP_CHAR   = u'¬'
UNK_CHAR = 'π'
MAX_ACTION_SEQ_LEN = 50
OPTIMIZERS = {'ADAM'    : lambda m: dy.AdamTrainer(m, lam=0.0, alpha=0.0001,
                                                   beta_1=0.9, beta_2=0.999, eps=1e-8),
            'SGD'     : dy.SimpleSGDTrainer,
            'ADADELTA': dy.AdadeltaTrainer}


### IO handling and evaluation

def check_path(path, arg_name, is_data_path=True):
    if not os.path.exists(path):
        prefix = DATA_PATH if is_data_path else RESULTS_PATH
        tmp = os.path.join(prefix, path)
        print tmp
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
        loss, prediction, predicted_actions = ti.transduce(input)
        if prediction == output:
            correct += 1
        final_results.append((input,prediction))  # pred expected as list
    accuracy = correct / test_data.length
    return accuracy, final_results

def write_results_file(hyper_params, accuracy, train_path, test_path, output_file_path, final_results):
    
    # write hyperparams, micro + macro avg. accuracy
    with codecs.open(output_file_path, 'w', encoding='utf8') as f:
        f.write('train path = ' + str(train_path) + '\n')
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

def log_to_file(log_file_name, e, avg_loss, train_accuracy, dev_accuracy):
    # if first write, add headers
    if e == 0:
        log_to_file(log_file_name, 'epoch', 'avg_loss', 'train_accuracy', 'dev_accuracy')
    
    with open(log_file_name, "a") as logfile:
        logfile.write("{}\t{}\t{}\t{}\n".format(e, avg_loss, train_accuracy, dev_accuracy))


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
        words = []
        with file(vocab_fname) as fh:
            for line in fh:
                line.strip()
                word, count = line.split()
                words.append(word)
        return Vocab.from_list(words)
    
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
    def __init__(self, model, train_data, arguments):
        
        self.INPUT_DIM    = int(arguments['--input'])
        self.HIDDEN_DIM   = int(arguments['--hidden'])
#        self.FEAT_INPUT_DIM = int(arguments['--feat-input'])
        self.LAYERS       = int(arguments['--layers'])
        self.dropout      = float(arguments['--dropout'])
        
        self.build_vocabularies(train_data)
        self.build_model(model)
        # for printing
        self.hyperparams = {'INPUT_DIM'       : self.INPUT_DIM,
                            'HIDDEN_DIM'      : self.HIDDEN_DIM,
                            #'FEAT_INPUT_DIM'  : self.FEAT_INPUT_DIM,
                            'LAYERS'          : self.LAYERS,
                            'DROPOUT'         : self.dropout}

    def build_vocabularies(self, train_data):
        
        # ACTION VOCABULARY
        acts =list(set([c for w in train_data.inputs for c in w] + [c for w in train_data.outputs for c in w])) + [STOP_CHAR] + [UNK_CHAR] + [BEGIN_CHAR]
        self.vocab_acts = Vocab.from_list(acts)
        
        self.BEGIN   = self.vocab_acts.w2i[BEGIN_CHAR]
        self.STOP   = self.vocab_acts.w2i[STOP_CHAR]
        self.UNK       = self.vocab_acts.w2i[UNK_CHAR]
        # rest are INSERT_* actions
        INSERT_CHARS, INSERTS = zip(*[(a, a_id) for a, a_id in self.vocab_acts.w2i.iteritems()
                                      if a not in set([BEGIN_CHAR, STOP_CHAR, UNK_CHAR])])
            
        self.INSERT_CHARS, self.INSERTS = list(INSERT_CHARS), list(INSERTS)
        self.NUM_ACTS = self.vocab_acts.size()
        print u'{} actions of which {} are INSERT actions: {}'.format(self.NUM_ACTS, len(self.INSERTS),
                                                                    u', '.join(self.INSERT_CHARS))

    def build_model(self, model):
        
        # BiLSTM for input
        self.fbuffRNN  = dy.CoupledLSTMBuilder(self.LAYERS, self.INPUT_DIM, self.HIDDEN_DIM, model)
        self.bbuffRNN  = dy.CoupledLSTMBuilder(self.LAYERS, self.INPUT_DIM, self.HIDDEN_DIM, model)
        
        # embedding lookups for actions
        self.ACT_LOOKUP  = model.add_lookup_parameters((self.NUM_ACTS, self.INPUT_DIM))

        # decoder state to hidden
        in_dim = self.INPUT_DIM

        self.decoder = dy.CoupledLSTMBuilder(self.LAYERS, in_dim, self.HIDDEN_DIM, model)

        # softmax parameters
        self.R = model.add_parameters((self.NUM_ACTS, 3 * self.HIDDEN_DIM))
        self.bias = model.add_parameters(self.NUM_ACTS)
        
        # attention MLPs - Loung-style with extra v_a from Bahdanau
        
        # concatenation layer for h (hidden dim), c (2 * hidden_dim)
        self.W_c = model.add_parameters((3 * self.HIDDEN_DIM, 3 * self.HIDDEN_DIM))
        
        # attention MLP's - Bahdanau-style
        # concatenation layer for h_input (2*hidden_dim), h_output (hidden_dim)
        self.W__a = model.add_parameters((self.HIDDEN_DIM, self.HIDDEN_DIM))
        
        # concatenation layer for h (hidden dim), c (2 * hidden_dim)
        self.U__a = model.add_parameters((self.HIDDEN_DIM, 2 * self.HIDDEN_DIM))
        
        # concatenation layer for h_input (2*hidden_dim), h_output (hidden_dim)
        self.v__a = model.add_parameters((1, self.HIDDEN_DIM))
        
        
        
        print 'Model dimensions:'
        print ' * ACTION EMBEDDING LAYER:    IN-DIM: {}, OUT-DIM: {}'.format(self.NUM_ACTS, self.INPUT_DIM)
        print
        print ' * ENCODER biLSTM: IN-DIM: {}, OUT-DIM: {}'.format(2*self.INPUT_DIM, 2*self.HIDDEN_DIM)
        print ' * DECODER LSTM:               IN-DIM: {}, OUT-DIM: {}'.format(in_dim, self.HIDDEN_DIM)
        print ' All LSTMs have {} layer(s)'.format(self.LAYERS)
        print
        print ' * SOFTMAX:                   IN-DIM: {}, OUT-DIM: {}'.format(self.HIDDEN_DIM, self.NUM_ACTS)
    

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

    def transduce(self, input, _oracle_actions=None, feats=None):
        
        # encode of oracle actions
        if _oracle_actions:
            oracle_actions = [self.vocab_acts.w2i[a] for a in _oracle_actions]
            oracle_actions += [self.STOP]
            oracle_actions = list(reversed(oracle_actions))
        
        R = dy.parameter(self.R)   # hidden to action
        bias = dy.parameter(self.bias)
        W_c = dy.parameter(self.W_c)
        W__a = dy.parameter(self.W__a)
        U__a = dy.parameter(self.U__a)
        v__a = dy.parameter(self.v__a)
        
        
        # biLSTM encoder
        input = BEGIN_CHAR + input + STOP_CHAR

        input_emb = []
        for char_ in reversed(input):
            char_id = self.vocab_acts.w2i.get(char_, self.UNK)
            char_embedding = self.ACT_LOOKUP[char_id]
            input_emb.append(char_embedding)
        biencoder = self.bilstm_transduce(self.fbuffRNN, self.bbuffRNN, input_emb)
        
        losses = []
        output = []
        action_history = [self.BEGIN] # <
        s = self.decoder.initial_state()
#        s=s0
        
        while not ((action_history[-1] == self.STOP) or len(action_history) == MAX_ACTION_SEQ_LEN):
            # compute probability of each of the actions and choose an action
            # either from the oracle or if there is no oracle, based on the model
            
            
            # decoder next state
            prev_action_id = action_history[-1]
            s = s.add_input(self.ACT_LOOKUP[prev_action_id])
            
            # soft attention vector
            scores = [v__a * dy.tanh(W__a * s.output() + U__a * h_input) for h_input in biencoder]
            alphas = dy.softmax(dy.concatenate(scores))
            c = dy.esum([h_input * dy.pick(alphas, j) for j, h_input in enumerate(biencoder)])
            
            # softmax over actions
            h_output = dy.tanh(W_c * dy.concatenate([s.output(), c]))
            probs = dy.softmax(R * h_output + bias)

            if _oracle_actions is None:
                action = np.argmax(probs.npvalue())
            else:
                action = oracle_actions.pop()
                
            losses.append(-dy.log(dy.pick(probs, action)))
            action_history.append(action)
            
            # execute the action to update the transducer state
            if action == self.STOP:
                break
            else:
                # one of the inserts
                insert_char = self.vocab_acts.i2w.get(action,UNK_CHAR)
                output.append(insert_char)

        output = u''.join(output)
        action_history = u''.join([self.vocab_acts.i2w[a] for a in action_history])
        return ((dy.average(losses) if losses else None), output, action_history)

if __name__ == "__main__":
    arguments = docopt(__doc__)
    print arguments
    
    np.random.seed(17)
    random.seed(17)

    train_path        = check_path(arguments['TRAIN_PATH'], 'TRAIN_PATH')
    dev_path          = check_path(arguments['DEV_PATH'], 'DEV_PATH')
    results_file_path = check_path(arguments['RESULTS_PATH'], 'RESULTS_PATH', is_data_path=False)

    # some filenames defined from `results_file_path`
    log_file_name   = results_file_path + '/log.txt'
    tmp_model_path  = results_file_path + '/bestmodel.txt'

    if arguments['--test_path']:
        test_path = check_path(arguments['--test_path'], 'test_path')
    else:
        # indicates no test set eval should be performed
        test_path = None

    print 'Train path: {}'.format(train_path)
    print 'Dev path: {}'.format(dev_path)
    print 'Results path: {}'.format(results_file_path)
    print 'Test path: {}'.format(test_path)


    print 'Loading data...'
    data_set = SoftDataSet

    train_data = data_set.from_file(train_path)
    dev_data = data_set.from_file(dev_path)
    if test_path:
        test_data = data_set.from_file(test_path)
    else:
        test_data = None

    print 'Checking if any special symbols in data...'
    for data, name in [(train_data, 'train'), (dev_data, 'dev')] + \
        ([(test_data, 'test')] if test_data else []):
        data = set(data.inputs + data.outputs)
        for c in [STOP_CHAR, UNK_CHAR]:
            assert c not in data
        print '{} data does not contain special symbols'.format(name)


    print 'Building model...'
    model = dy.Model()
    ti = SoftAttention(model, train_data, arguments)
        
    # Training hypoparameters
    optimization = arguments['--optimization']
    epochs = int(arguments['--epochs'])
    max_patience = int(arguments['--patience'])

    hyperparams = {'MAX_ACTION_SEQ_LEN': MAX_ACTION_SEQ_LEN,
                   'OPTIMIZATION': optimization,
                   'EPOCHS': epochs,
                   'PATIENCE': max_patience,
                   'BEAM_WIDTH': 1}
    # Model hypoparameters
    for k, v in ti.hyperparams.items():
        hyperparams[k] = v

    for k, v in hyperparams.items():
        print '{:20} = {}'.format(k, v)
    print

    if not arguments['--eval']:
        # perform model training
        trainer = OPTIMIZERS.get(optimization, OPTIMIZERS['SGD'])
        print 'Using {} trainer: {}'.format(optimization, trainer)
        trainer = trainer(model)

        total_loss = 0.  # total train loss that is...
        best_avg_dev_loss = 999.
        best_dev_accuracy = -1.
        best_train_accuracy = -1.
        train_len = train_data.length
        dev_len = dev_data.length
        sanity_set_size = 100 # for speed
        patience = 0
        previous_predicted_actions = [[None]*sanity_set_size]

        # progress bar init
        widgets = [progressbar.Bar('>'), ' ', progressbar.ETA()]
        train_progress_bar = progressbar.ProgressBar(widgets=widgets, maxval=epochs).start()
        avg_loss = -1  # avg training loss that is...

        # does not change from epoch to epoch due to re-shuffling
        dev_set = dev_data.iter()

        for epoch in xrange(epochs):
            print 'training...'
            then = time.time()

            # compute loss for each sample and update
            for i, (input, output) in enumerate(train_data.iter(shuffle=True)):
                # here we do training
                dy.renew_cg()
                loss, _, _ = ti.transduce(input, output)
                if loss is not None:
                    total_loss += loss.scalar_value()
                    loss.backward()
                    trainer.update()
                if i > 0:
                    avg_loss = total_loss / (i + epoch * train_len)
                else:
                    avg_loss = total_loss

            print '\t...finished in {:.3f} sec'.format(time.time() - then)

            # condition for displaying stuff like true outputs and predictions
            check_condition = (epoch > 0 and (epoch % 5 == 0 or epoch == epochs - 1))

            # get train accuracy
            print 'evaluating on train...'
            then = time.time()
            train_correct = 0.
            pred_acts = []
            for i, (input, output) in enumerate(train_data.iter(indices=sanity_set_size)):
                _, prediction, predicted_actions = ti.transduce(input)

                if prediction == output:
                    train_correct += 1

            train_accuracy = train_correct / sanity_set_size
            print '\t...finished in {:.3f} sec'.format(time.time() - then)

            if train_accuracy > best_train_accuracy:
                best_train_accuracy = train_accuracy

            # get dev accuracy
            print 'evaluating on dev...'
            then = time.time()
            dev_correct = 0.
            dev_loss = 0.
            for input, output in dev_set:
                loss, prediction, predicted_actions = ti.transduce(input)
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
            dev_accuracy = dev_correct / dev_len
            print '\t...finished in {:.3f} sec'.format(time.time() - then)

            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                #then = time.time()
                # save best model to disk
                model.save(tmp_model_path)
                print 'saved new best model to {}'.format(tmp_model_path)
                #print '\t...finished in {:.3f} sec'.format(time.time() - then)
                patience = 0
            else:
                patience += 1

            # found "perfect" model
            if dev_accuracy == 1:
                train_progress_bar.finish()
                break

            # get dev loss
            avg_dev_loss = dev_loss / dev_len

            if avg_dev_loss < best_avg_dev_loss:
                best_avg_dev_loss = avg_dev_loss

            print ('epoch: {0} train loss: {1:.4f} dev loss: {2:.4f} dev accuracy: {3:.4f} '
                   'train accuracy: {4:.4f} best dev accuracy: {5:.4f} best train accuracy: {6:.4f} '
                   'patience = {7}').format(epoch, avg_loss, avg_dev_loss, dev_accuracy, train_accuracy,
                                            best_dev_accuracy, best_train_accuracy, patience)

            log_to_file(log_file_name, epoch, avg_loss, train_accuracy, dev_accuracy)

            if patience == max_patience:
                print 'out of patience after {} epochs'.format(epoch)
                train_progress_bar.finish()
                break
            # finished epoch
            train_progress_bar.update(epoch)
        print 'finished training. average loss: {}'.format(avg_loss)

    else:
        print 'skipped training by request. evaluating best models.'

    # eval on dev
    print '=========DEV EVALUATION:========='
    model = dy.Model()
    ti = SoftAttention(model, train_data, arguments)

    print 'trying to load model from: {}'.format(tmp_model_path)
    model.populate(tmp_model_path)

    accuracy, dev_results = get_accuracy_predictions(ti, dev_data)
    print 'accuracy: {}'.format(accuracy)
    output_file_path = results_file_path + '/best.dev'

    write_results_file(hyperparams, accuracy, train_path, dev_path, output_file_path, dev_results)

    if test_data:
        # eval on test
        print '=========TEST EVALUATION:========='
        accuracy, test_results = get_accuracy_predictions(ti, test_data)
        print 'accuracy: {}'.format(accuracy)
        output_file_path = results_file_path + '/best.test'

        write_results_file(hyperparams, accuracy, train_path, test_path, output_file_path, test_results)
