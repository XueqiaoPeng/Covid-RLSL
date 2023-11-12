import torch
from torch import nn
import pandas as pd
import numpy as np
import time
import os
import os.path as osp
import joblib
import matplotlib.pyplot as plt
from spinup import EpochLogger

# '''data generation'''
def load_policy_and_env(fpath, itr='last', deterministic=False):

    # determine if tf save or pytorch save
    backend = 'pytorch'
    # handle which epoch to load from
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        backend == 'pytorch'
        pytsave_path = osp.join(fpath, 'pyt_save')
        # Each file in this folder has naming convention 'modelXX.pt', where
        # 'XX' is either an integer or empty string. Empty string case
        # corresponds to len(x)==8, hence that case is excluded.
        saves = [int(x.split('.')[0][3:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]

        itr = '%d'%max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    get_action = load_pytorch_policy(fpath, itr, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""
    
    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    # print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    return get_action

if __name__ == '__main__':
    # """evaluation"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str)
    parser.add_argument('--dpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=30)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action = load_policy_and_env(args.fpath, 
                                          args.itr if args.itr >=0 else 'last',
                                          args.deterministic)
    logger = EpochLogger()
    o, r, d, ep_ret, ep_len,n = env.reset(), 0, False, 0, 0,0
    info = np.zeros(9)
    clf2 = joblib.load(args.dpath)
    while n < 50:
        state = info[0:8].reshape(1, -1)
        a = clf2.predict(state).astype(int)
        o, r, d, info = env.step(a)
        # print(state)
        # print(a)
        # print(r)
        ep_ret += r
        ep_len += 1
        if d or (ep_len == args.len):
            logger.store(EpRet= ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d\t'%(n,  ep_ret, ep_len))
            # print('EvalRet %d'%(reward))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1
    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()
    

        