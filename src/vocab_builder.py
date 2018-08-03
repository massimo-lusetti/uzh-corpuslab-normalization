#!/usr/bin/env python
# -*- coding: utf-8 -*

from common import BEGIN_CHAR,STOP_CHAR,UNK_CHAR
import codecs
import collections

# represents a bidirectional mapping from strings to ints
class Vocab(object):
    def __init__(self, w2i):
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.iteritems()}
    
    def save(self, vocab_path):
        with codecs.open(vocab_path, 'w', 'utf-8') as fh:
            for w,i in sorted(self.w2i.iteritems(),key=lambda v:v[0]):
                fh.write(u'{}\t{}\n'.format(w,i))
        return

    
    @classmethod
    def from_list(cls, words, w2i=None):
        if w2i:
            idx=len(w2i)
        else:
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
                word, idx = line.rstrip().split('\t')
                w2i[word] = int(idx)
                #print word, idx
        return Vocab(w2i)
    
    def size(self): return len(self.w2i.keys())

def build_vocabulary(train_data, vocab_path, vocab_trunk=0):
    # Build vocabulary over items - chars or segments - and save it to 'vocab_path'
    
    if vocab_trunk==0:
        items = list(set([c for w in train_data for c in w])) #+ [STOP_CHAR] + [UNK_CHAR] + [BEGIN_CHAR]
        print set([c for w in train_data for c in w])
    else:
        tokens = [c for w in train_data for c in w]
        counter = collections.Counter(tokens)
        print u'Word types in train set: {}'.format(len(counter))
        n = len(counter) - int(len(counter)*vocab_trunk)
        print u'Trunkating: {}'.format(n)
        items = [w for w,c in counter.most_common(n)]


    # to make sure that special symbols have the same index across models
    w2i = {}
    w2i[BEGIN_CHAR] = 0
    w2i[STOP_CHAR] = 1
    w2i[UNK_CHAR] = 2
    print 'Vocabulary size: {}'.format(len(items))
    print 'Example of vocabulary items:' + u', '.join(items[:10])
    print
    vocab = Vocab.from_list(items,w2i)
    vocab.save(vocab_path)
    return
