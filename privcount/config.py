'''
Created on Dec 8, 2016

@author: teor

Based on previous code
'''

from os import path

def normalise_path(path_str):
    '''
    Return the abolute path corresponding to path_str, with user directories
    expanded, and the current working directory assumed for relative paths
    '''
    expanded_path = path.expanduser(path_str)
    return path.abspath(expanded_path)

def choose_secret_handshake_path(local_conf, global_conf):
    '''
    Determine the secret handshake path using the first path from:
    - local_conf,
    - global_conf, or
    - the default hard-coded path,
    and return that path.
    '''
    if 'secret_handshake' in local_conf:
        return normalise_path(local_conf['secret_handshake'])
    # unlike other top-level configs, this is a file path, not data
    elif 'secret_handshake' in global_conf:
        return normalise_path(global_conf['secret_handshake'])
    # if the path is not specified, use the default path
    else:
        return normalise_path('privcount.secret_handshake.yaml')
