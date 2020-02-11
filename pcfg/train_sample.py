#!/usr/bin/env python3
import sys
import os

import argparse
import json
import random
import shutil
import copy

import torch
from torch import cuda
import numpy as np
import time
import logging
from data import Dataset
from utils import *
from models import NeuralPCFG
import pickle

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--data_file', default='pcfg-data.pkl')
parser.add_argument('--save_path', default='sample-neural-pcfg.pt', help='where to save the model')
parser.add_argument('--train_size', default=5000, type=int)

# Model options
# Generative model parameters
parser.add_argument('--use_scalar', default=0, type=int)
parser.add_argument('--t_states', default=10, type=int, help='number of preterminal states')
parser.add_argument('--nt_states', default=10, type=int, help='number of nonterminal states')
parser.add_argument('--state_dim', default=256, type=int, help='symbol embedding dimension')
# Optimization options
parser.add_argument('--num_epochs', default=10, type=int, help='number of training epochs')
parser.add_argument('--lr', default=0.001, type=float, help='starting learning rate')
parser.add_argument('--max_grad_norm', default=3, type=float, help='gradient clipping parameter')
parser.add_argument('--max_length', default=40, type=float, help='max sentence length cutoff start')
parser.add_argument('--param_init', default=0.1, type=float)
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--seed', default=3435, type=int, help='random seed')
parser.add_argument('--print_every', type=int, default=1000, help='print stats after N batches')

def main(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  data = pickle.load(open(args.data_file, "rb"))  

  train_data = data["train"]
  val_data = data["dev"]
  test_data = data["test"]
  true_dist = data["pcfg_probs"]
  vocab_size = data["vocab_size"]
  train_data = train_data[:args.train_size]
  cuda.set_device(args.gpu)
  model = NeuralPCFG(vocab = vocab_size,
                     state_dim = args.state_dim,
                     t_states = args.t_states,
                     nt_states = args.nt_states,
                     use_scalar = args.use_scalar)
  for name, param in model.named_parameters():    
    param.data.uniform_(-args.param_init, args.param_init)
  model.cuda()
  model.train()
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  best_val_nll = 1e5
  best_train_nll = 1e5
  epoch = 0
    
  while epoch < args.num_epochs:
    start_time = time.time()
    epoch += 1  
    print('Starting epoch %d' % epoch)
    train_nll = 0.
    num_sents = 0.
    num_words = 0.
    b = 0
    for i in np.random.permutation(len(train_data)):
      b += 1
      sents = train_data[i]
      length = len(train_data[i])
      if length > args.max_length or length == 1: 
        continue
      sents = torch.LongTensor(sents).unsqueeze(0)
      sents = sents.cuda()
      optimizer.zero_grad()
      nll = model(sents)
      nll.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)      
      optimizer.step()
      train_nll += nll.sum().item()
      num_sents += 1
      num_words += length
      if b % args.print_every == 0:
        param_norm = sum([p.norm()**2 for p in model.parameters()]).item()**0.5
        gparam_norm = sum([p.grad.norm()**2 for p in model.parameters() 
                           if p.grad is not None]).item()**0.5
        log_str = 'Symbols: %d, Epoch: %d, Batch: %d/%d, |Param|: %.6f, |GParam|: %.2f,  LR: %.4f, ' + \
                  'TrainNLL: %.3f, ValNLL: %.3f, ' + \
                  'Throughput: %.2f examples/sec'
        print(log_str %
              (model.nt_states, epoch, b, len(train_data), param_norm, gparam_norm, args.lr, 
               train_nll / num_words, best_val_nll, 
               num_sents / (time.time() - start_time)))
    print('--------------------------------')
    print('Checking validation perf...')    
    val_nll = eval(val_data, model)
    print('--------------------------------')
    if val_nll < best_val_nll:
      best_val_nll = val_nll
      print("Best model achieved")
      print("Train: %.3f, Val: %3f" % (train_nll / num_words, best_val_nll))
      checkpoint = {
        'args': args.__dict__,
        'model': model.cpu(),
      }
      print('Saving checkpoint to %s' % args.save_path)
      torch.save(checkpoint, args.save_path)
      model.cuda()

def eval(data, model):
  model.eval()
  num_sents = 0
  num_words = 0
  total_nll = 0.
  with torch.no_grad():
    for i in range(len(data)):
      sents = data[i]
      length = len(data[i])
      if length > args.max_length or length == 1: #length filter based on curriculum 
        continue
      sents = torch.LongTensor(sents).unsqueeze(0)
      sents = sents.cuda()
      nll = model(sents)
      total_nll += nll.sum().item()
      num_sents += 1
      num_words += length
  nll = total_nll / num_words
  print('NLL: %.3f' % (nll))
  model.train()
  return nll

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
