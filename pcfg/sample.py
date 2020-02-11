#!/usr/bin/env python3
import sys
import os

import argparse
import json
import random
import shutil
import copy
import pickle
import torch
from torch import cuda
import numpy as np
import PCFG
from models import NeuralPCFG
import torch.distributions
from torch.distributions.dirichlet import Dirichlet

parser = argparse.ArgumentParser()


parser.add_argument('--train_samples', default=50000, type=int)
parser.add_argument('--dev_samples', default=1000, type=int)
parser.add_argument('--test_samples', default=1000, type=int)
parser.add_argument('--model_file', default='')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--seed', default=3435, type=int, help='random seed')

parser.add_argument('--output_file', default='pcfg-data.pkl')

def main(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  print('loading model from ' + args.model_file)
  checkpoint = torch.load(args.model_file)
  model = checkpoint['model']
  cuda.set_device(args.gpu)
  model.eval()
  model.cuda()
  
  s_dist, nt_dist, pt_dist = model.get_rule_probs()
  print("Sampling %d train, %d dev, %d test sentences..." % 
        (args.train_samples, args.dev_samples, args.test_samples))
  data, freqs = PCFG.sample(s_dist, nt_dist, pt_dist, 
                     args.train_samples + args.dev_samples + args.test_samples)
  print("Done!")
  lengths = np.array([len(d) for d in data])
  print("Length statistics")
  print("Min: %d, Mean: %d, Max: %d" % 
        (np.min(lengths), np.mean(lengths), np.max(lengths)))
  print("50pct: %d, 75pct: %d, 90pct: %d, 95pct: %d, 99pct: %d" % 
        (np.percentile(lengths, 50),
         np.percentile(lengths, 75),
         np.percentile(lengths, 90),
         np.percentile(lengths, 95),
         np.percentile(lengths, 99)))
  train = data[:args.train_samples]
  dev = data[args.train_samples:args.train_samples + args.dev_samples]
  test = data[args.train_samples + args.dev_samples:]
  print("Train: %d, Dev: %d, Test: %d" % (len(train), len(dev), len(test)))
  
  output = {'vocab_size': model.vocab,
            'nt_states': model.nt_states,
            't_states': model.t_states,
            'pcfg_probs': [s_dist, nt_dist, pt_dist],
            'train': train,
            'dev': dev,
            'test': test,
            'args': args.__dict__}
  
  pickle.dump(output, open(args.output_file, "wb"))

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
