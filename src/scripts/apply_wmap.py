"""This script applies a word map to sentences in stdin. If --dir is
set to s2i, the word strings in stdin are converted to their ids. If
--dir is i2s, we convert word IDs to their readable representations.
"""

import logging
import argparse
import sys
import codecs

def load_wmap(path, inverse=False):
    with codecs.open(path, 'r', 'utf8') as f:
        d = dict(line.rstrip().split('\t') for line in f)
        #if inverse:
        #d = dict(zip(d.values(), d.keys()))
#        for (s, i) in [('<s>', '1'), ('</s>', '2')]:
        for (s, i) in [('<s>', '0'), ('</s>', '1')]:
            if not s in d or d[s] != i:
                logging.warning("%s has not ID %s in word map %s!" % (s, i, path))
        #print(d)
        return d

parser = argparse.ArgumentParser(description='Convert between written and ID representation of words. '
                                 'The index 0 is always used as UNK token, wmap entry for 0 is ignored. '
                                 'Usage: python apply_wmap.py < in_sens > out_sens')
parser.add_argument('-d','--dir', help='s2i: convert to IDs (default), i2s: convert from IDs',
                    required=False)
parser.add_argument('-m','--wmap', help='Word map to apply (format: see -i parameter)',
                    required=True)
parser.add_argument('-i','--inverse_wmap', help='Use this argument to use word maps with format "id word".'
                    ' Otherwise the format "word id" is assumed', action='store_true')
parser.add_argument('--low', help='Lowercase', action='store_true')
args = parser.parse_args()


def process():

    wmap = load_wmap(args.wmap, args.inverse_wmap)

    unk = '0'
    if args.dir and args.dir == 'i2s': # inverse wmap
        wmap = dict(zip(wmap.values(), wmap.keys()))
        unk = "NOTINWMAP"

    while True:
        line = sys.stdin.readline().decode('utf-8').lower() if args.low else sys.stdin.readline().decode('utf-8')
        if not line: break # EOF
        print(' '.join([wmap[w] if (w in wmap) else unk for w in line.rstrip().replace('   ','|').replace(' ','').replace('|',' ')])) #strip().split()]))

if __name__ == "__main__":
    process()

