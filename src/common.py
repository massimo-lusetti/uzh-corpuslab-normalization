#!/usr/bin/env python
# -*- coding: utf-8 -*

import os
# Default paths
SRC_FOLDER = os.path.dirname(__file__)
RESULTS_FOLDER = os.path.join(SRC_FOLDER, '../results')
DATA_FOLDER = os.path.join(SRC_FOLDER, '../data/')



# Model defaults
BEGIN_CHAR   = u'<s>'
STOP_CHAR   = u'</s>'
UNK_CHAR = u'<unk>'
BOUNDARY_CHAR = u' '

### IO handling and evaluation

def check_path(path, arg_name, is_data_path=True): #common
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
                print tmp
                os.makedirs(tmp)
                path = tmp
    return path

