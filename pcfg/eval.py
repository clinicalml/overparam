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
parser.add_argument('--model_file', default='sample-neural-pcfg.pt')
parser.add_argument('--max_length', default=40, type=int)
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--seed', default=3435, type=int, help='random seed')

def main(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  data = pickle.load(open(args.data_file, "rb"))  
  checkpoint = torch.load(args.model_file)
  model = checkpoint['model']
  cuda.set_device(args.gpu)
  model.eval()
  model.cuda()

  train_data = data["train"]
  val_data = data["dev"]
  test_data = data["test"]
  true_dist = data["pcfg_probs"]
  vocab_size = data["vocab_size"]  
  true_model = NeuralPCFG(vocab = vocab_size,
                          state_dim = 256,
                          t_states = data["t_states"],
                          nt_states = data["nt_states"],
                          use_scalar = 1)  
  true_model.root_logits.data.copy_(true_dist[0].data.log())
  true_model.rule_logits.data.copy_(true_dist[1].data.log())
  true_model.emission_logits.data.copy_(true_dist[2].data.log())
  true_model.cuda().eval()
  print("Evaluating...")
  eval(test_data, model, true_model)

def eval(data, model, true_model):
  model.eval()
  num_sents = 0
  num_words = 0
  total_nll = 0.
  corpus_f1 = [0., 0., 0.] 
  nt_freq = {}
  t_freq = {}
  with torch.no_grad():
    for i in range(len(data)):
      sents = data[i]
      length = len(data[i])
      sents = torch.LongTensor(sents).unsqueeze(0)
      sents = sents.cuda()
      true_nll, true_binary_matrix, true_argmax_spans = true_model(sents, argmax=True)
      if length > args.max_length or length == 1:
        continue
      nll, binary_matrix, argmax_spans = model(sents, argmax=True)
      total_nll += nll.sum().item()
      pred_span = set([(a[0], a[1]) for a in argmax_spans[0]])    
      gold_span = set([(a[0], a[1]) for a in true_argmax_spans[0]])
      tp, fp, fn = get_stats(pred_span, gold_span) 
      corpus_f1[0] += tp
      corpus_f1[1] += fp
      corpus_f1[2] += fn

      num_sents += 1
      num_words += length
  tp, fp, fn = corpus_f1  
  prec = tp / (tp + fp)
  recall = tp / (tp + fn)
  f1 = 2*prec*recall/(prec+recall) if prec+recall > 0 else 0.
  nll = total_nll / num_words
  ppl = np.exp(nll)
  print("Test results")
  print('NLL: %.3f, F1: %.3f' % (nll, f1))
  return nll, f1

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
