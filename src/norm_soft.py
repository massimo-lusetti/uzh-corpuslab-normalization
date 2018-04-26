#!/usr/bin/env python
# -*- coding: utf-8 -*
"""Trains encoder-decoder model with soft attention.

Usage:
  norm_soft.py train [--dynet-seed SEED] [--dynet-mem MEM]
    [--input=INPUT] [--hidden=HIDDEN] [--feat-input=FEAT] [--layers=LAYERS] [--vocab_path=VOCAB_PATH]
    [--dropout=DROPOUT] [--epochs=EPOCHS] [--patience=PATIENCE] [--optimization=OPTIMIZATION]
    MODEL_FOLDER --train_path=TRAIN_FILE --dev_path=DEV_FILE
  norm_soft.py test [--dynet-mem MEM]
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
  --vocab_path=VOCAB_PATH       vocab path, possibly relative to RESULTS_FOLDER [default: vocab.txt]
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

from common import BEGIN_CHAR,STOP_CHAR,UNK_CHAR, SRC_FOLDER,RESULTS_FOLDER,DATA_FOLDER,check_path, write_pred_file, write_param_file, write_eval_file
from vocab_builder import build_vocabulary, Vocab

MAX_PRED_SEQ_LEN = 50 # option
OPTIMIZERS = {'ADAM'    : lambda m: dy.AdamTrainer(m, lam=0.0, alpha=0.0001, #common
                                                   beta_1=0.9, beta_2=0.999, eps=1e-8),
    'SGD'     : dy.SimpleSGDTrainer,
        'ADADELTA': dy.AdadeltaTrainer}


### IO handling and evaluation

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

def log_to_file(log_file_name, e, avg_train_loss, train_accuracy, dev_accuracy):
    # if first write, add headers
    if e == 0:
        log_to_file(log_file_name, 'epoch', 'avg_train_loss', 'train_accuracy', 'dev_accuracy')
    
    with open(log_file_name, "a") as logfile:
        logfile.write("{}\t{}\t{}\t{}\n".format(e, avg_train_loss, train_accuracy, dev_accuracy))

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
    def __init__(self, pc, model_hyperparams, best_model_path=None):
        
        self.hyperparams = model_hyperparams
        
        print 'Loading vocabulary from {}:'.format(self.hyperparams['VOCAB_PATH'])
        self.vocab = Vocab.from_file(self.hyperparams['VOCAB_PATH'])
        self.BEGIN   = self.vocab.w2i[BEGIN_CHAR]
        self.STOP   = self.vocab.w2i[STOP_CHAR]
        self.UNK       = self.vocab.w2i[UNK_CHAR]
        self.hyperparams['VOCAB_SIZE'] = self.vocab.size()
        
        self.build_model(pc, best_model_path)
        
        print 'Model Hypoparameters:'
        for k, v in self.hyperparams.items():
            print '{:20} = {}'.format(k, v)
        print
        
    def build_model(self, pc, best_model_path):
        
        if best_model_path:
            print 'Loading model from: {}'.format(best_model_path)
            self.fbuffRNN, self.bbuffRNN, self.VOCAB_LOOKUP, self.decoder, self.R, self.bias, self.W_c, self.W__a, self.U__a,  self.v__a = dy.load(best_model_path, pc)
        else:
            # BiLSTM for input
            self.fbuffRNN  = dy.CoupledLSTMBuilder(self.hyperparams['LAYERS'], self.hyperparams['INPUT_DIM'], self.hyperparams['HIDDEN_DIM'], pc)
            self.bbuffRNN  = dy.CoupledLSTMBuilder(self.hyperparams['LAYERS'], self.hyperparams['INPUT_DIM'], self.hyperparams['HIDDEN_DIM'], pc)
            
            # embedding lookups for vocabulary
            self.VOCAB_LOOKUP  = pc.add_lookup_parameters((self.hyperparams['VOCAB_SIZE'], self.hyperparams['INPUT_DIM']))

            # decoder LSTM
            self.decoder = dy.CoupledLSTMBuilder(self.hyperparams['LAYERS'], self.hyperparams['INPUT_DIM'], self.hyperparams['HIDDEN_DIM'], pc)

            # softmax parameters
            self.R = pc.add_parameters((self.hyperparams['VOCAB_SIZE'], 3 * self.hyperparams['HIDDEN_DIM']))
            self.bias = pc.add_parameters(self.hyperparams['VOCAB_SIZE'])
            
            # attention MLPs - Loung-style with extra v_a from Bahdanau
            
            # concatenation layer for h (hidden dim), c (2 * hidden_dim)
            self.W_c = pc.add_parameters((3 * self.hyperparams['HIDDEN_DIM'], 3 * self.hyperparams['HIDDEN_DIM']))
            
            # attention MLP's - Bahdanau-style
            # concatenation layer for h_input (2*hidden_dim), h_output (hidden_dim)
            self.W__a = pc.add_parameters((self.hyperparams['HIDDEN_DIM'], self.hyperparams['HIDDEN_DIM']))
            
            # concatenation layer for h (hidden dim), c (2 * hidden_dim)
            self.U__a = pc.add_parameters((self.hyperparams['HIDDEN_DIM'], 2 * self.hyperparams['HIDDEN_DIM']))
            
            # concatenation layer for h_input (2*hidden_dim), h_output (hidden_dim)
            self.v__a = pc.add_parameters((1, self.hyperparams['HIDDEN_DIM']))
        
        
        print 'Model dimensions:'
        print ' * VOCABULARY EMBEDDING LAYER: IN-DIM: {}, OUT-DIM: {}'.format(self.hyperparams['VOCAB_SIZE'], self.hyperparams['INPUT_DIM'])
        print
        print ' * ENCODER biLSTM: IN-DIM: {}, OUT-DIM: {}'.format(2*self.hyperparams['INPUT_DIM'], 2*self.hyperparams['HIDDEN_DIM'])
        print ' * DECODER LSTM: IN-DIM: {}, OUT-DIM: {}'.format(self.hyperparams['INPUT_DIM'], self.hyperparams['HIDDEN_DIM'])
        print ' All LSTMs have {} layer(s)'.format(self.hyperparams['LAYERS'])
        print
        print ' * SOFTMAX: IN-DIM: {}, OUT-DIM: {}'.format(self.hyperparams['HIDDEN_DIM'], self.hyperparams['VOCAB_SIZE'])
        print

    def save_model(self, best_model_path):
        dy.save(best_model_path, [self.fbuffRNN, self.bbuffRNN, self.VOCAB_LOOKUP, self.decoder, self.R, self.bias, self.W_c, self.W__a, self.U__a,  self.v__a])


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
            try:
                true_output = [self.vocab.w2i[a] for a in _true_output]
            except:
                print a
                print _true_output
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

    def evaluate(self, data):
        # data is a list of tuples (an instance of SoftDataSet with iter method applied)
        total_loss = 0.
        correct = 0.
        final_results = []
        for input,output in data:
            loss, prediction = self.transduce(input)
            total_loss += loss.scalar_value()
            if prediction == output:
                correct += 1
            final_results.append((input,prediction))  # pred expected as list
        avg_loss = total_loss/len(data)
        accuracy = correct / len(data)
        return accuracy, final_results, avg_loss

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
        print 'Train data has {} examples'.format(train_data.length)
        dev_path = check_path(arguments['--dev_path'], 'dev_path')
        dev_data = data_set.from_file(dev_path)
        print 'Dev data has {} examples'.format(dev_data.length)
    
        print 'Checking if any special symbols in data...'
        for data, name in [(train_data, 'train'), (dev_data, 'dev')]:
            data = set(data.inputs + data.outputs)
            for c in [BEGIN_CHAR, STOP_CHAR, UNK_CHAR]:
                assert c not in data
            print '{} data does not contain special symbols'.format(name)
        print
        
        vocab_path = os.path.join(model_folder,arguments['--vocab_path'])
        if not os.path.exists(vocab_path):
            print 'Building vocabulary..'
            data = set(train_data.inputs + train_data.outputs)
            build_vocabulary(data, vocab_path)

        # Paths for checks and results
        log_file_name   = model_folder + '/log.txt'
        best_model_path  = model_folder + '/bestmodel.txt'
        output_file_path = model_folder + '/best.dev'

        # Model hypoparameters
        model_hyperparams = {'INPUT_DIM': int(arguments['--input']),
                            'HIDDEN_DIM': int(arguments['--hidden']),
                            #'FEAT_INPUT_DIM': int(arguments['--feat-input']),
                            'LAYERS': int(arguments['--layers']),
                            'VOCAB_PATH': vocab_path}
    
        print 'Building model...'
        pc = dy.ParameterCollection()
        ti = SoftAttention(pc, model_hyperparams)

        # Training hypoparameters
        train_hyperparams = {'MAX_PRED_SEQ_LEN': MAX_PRED_SEQ_LEN,
                            'OPTIMIZATION': arguments['--optimization'],
                            'EPOCHS': int(arguments['--epochs']),
                            'PATIENCE': int(arguments['--patience']),
                            'DROPOUT': float(arguments['--dropout']),
                            'BEAM_WIDTH': 1,
                            'TRAIN_PATH': train_path,
                            'DEV_PATH': dev_path}

        print 'Train Hypoparameters:'
        for k, v in train_hyperparams.items():
            print '{:20} = {}'.format(k, v)
        print
        
        trainer = OPTIMIZERS[train_hyperparams['OPTIMIZATION']]
        trainer = trainer(pc)

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
            train_accuracy, _, _ = ti.evaluate(train_data.iter(indices=sanity_set_size))
            print '\t...finished in {:.3f} sec'.format(time.time() - then)
            
            # get dev accuracy
            print 'evaluating on dev...'
            then = time.time()
            dy.renew_cg() # new graph for all the examples
            dev_accuracy, _, avg_dev_loss = ti.evaluate(dev_data.iter())
            print '\t...finished in {:.3f} sec'.format(time.time() - then)

            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                # save best model
                ti.save_model(best_model_path)
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

            if patience == train_hyperparams['PATIENCE']:
                print 'out of patience after {} epochs'.format(epoch)
                train_progress_bar.finish()
                break
            # finished epoch
            train_progress_bar.update(epoch)
                
        print 'finished training.'
        
        dev_accuracy, dev_results, _ = ti.evaluate(dev_data.iter())
        print 'Best dev accuracy: {}'.format(dev_accuracy)
        write_param_file(output_file_path, dict(model_hyperparams.items()+train_hyperparams.items()))
        write_pred_file(output_file_path, dev_results)
        write_eval_file(output_file_path, best_dev_accuracy, dev_path)

    elif arguments['test']:
        print '=========EVALUATION ONLY:========='
        # requires test path, model path of pretrained path and results path where to write the results to
        assert arguments['--test_path']!=None

        print 'Loading data...'
        test_path = check_path(arguments['--test_path'], '--test_path')
        data_set = SoftDataSet
        test_data = data_set.from_file(test_path)
        print 'Test data has {} examples'.format(test_data.length)

        print 'Checking if any special symbols in data...'
        data = set(test_data.inputs + test_data.outputs)
        for c in [BEGIN_CHAR, STOP_CHAR, UNK_CHAR]:
            assert c not in data
        print 'Test data does not contain special symbols'

        best_model_path  = model_folder + '/bestmodel.txt'
        output_file_path = model_folder + '/best.test'
        hypoparams_file = model_folder + '/best.dev'
        
        hypoparams_file_reader = codecs.open(hypoparams_file, 'r', 'utf-8')
        hyperparams_dict = dict([line.strip().split(' = ') for line in hypoparams_file_reader.readlines()])
        model_hyperparams = {'INPUT_DIM': int(hyperparams_dict['INPUT_DIM']),
                            'HIDDEN_DIM': int(hyperparams_dict['HIDDEN_DIM']),
                            #'FEAT_INPUT_DIM': int(hyperparams_dict['FEAT_INPUT_DIM']),
                            'LAYERS': int(hyperparams_dict['LAYERS']),
                            'VOCAB_PATH': hyperparams_dict['VOCAB_PATH']}

        pc = dy.ParameterCollection()
        ti = SoftAttention(pc, model_hyperparams, best_model_path)

        print 'Evaluating on test..'
        accuracy, test_results, _ = ti.evaluate(test_data.iter())
        print 'accuracy: {}'.format(accuracy)
        write_pred_file(output_file_path, test_results)
        write_eval_file(output_file_path, accuracy, test_path)

