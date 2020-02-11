import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from PCFG import PCFG
from random import shuffle

class ResidualLayer(nn.Module):
  def __init__(self, in_dim = 100,
               out_dim = 100):
    super(ResidualLayer, self).__init__()
    self.lin1 = nn.Linear(in_dim, out_dim)
    self.lin2 = nn.Linear(out_dim, out_dim)

  def forward(self, x):
    return F.relu(self.lin2(F.relu(self.lin1(x)))) + x

class NeuralPCFG(nn.Module):
  def __init__(self, vocab = 100,
               state_dim = 256, 
               t_states = 10,
               nt_states = 10,
               use_scalar = 0):
    super(NeuralPCFG, self).__init__()
    self.vocab = vocab
    self.state_dim = state_dim
    self.t_emb = nn.Parameter(torch.randn(t_states, state_dim))
    self.nt_emb = nn.Parameter(torch.randn(nt_states, state_dim))
    self.root_emb = nn.Parameter(torch.randn(1, state_dim))
    self.pcfg = PCFG(nt_states, t_states)
    self.nt_states = nt_states
    self.t_states = t_states
    self.all_states = nt_states + t_states
    self.dim = state_dim
    self.register_parameter('t_emb', self.t_emb)
    self.register_parameter('nt_emb', self.nt_emb)
    self.register_parameter('root_emb', self.root_emb)
    self.rule_mlp = nn.Linear(state_dim, self.all_states**2)
    self.root_mlp = nn.Sequential(nn.Linear(state_dim, state_dim),
                                  ResidualLayer(state_dim, state_dim),
                                  ResidualLayer(state_dim, state_dim),                         
                                  nn.Linear(state_dim, self.nt_states))
    self.vocab_mlp = nn.Sequential(nn.Linear(state_dim, state_dim),
                                   ResidualLayer(state_dim, state_dim),
                                   ResidualLayer(state_dim, state_dim),
                                   nn.Linear(state_dim, vocab))
      
    self.root_logits = nn.Parameter(torch.randn(nt_states))
    self.rule_logits = nn.Parameter(torch.randn(nt_states, (nt_states+t_states)**2))
    self.emission_logits = nn.Parameter(torch.randn(t_states, vocab))
    self.register_parameter('root_logits', self.root_logits)
    self.register_parameter('rule_logits', self.rule_logits)
    self.register_parameter('emission_logits', self.emission_logits)
    self.use_scalar = use_scalar

  def forward(self, x, argmax=False, use_mean=False):
    #x : batch x n
    n = x.size(1)
    batch_size = x.size(0)
    x_expand = x.unsqueeze(2).expand(batch_size, x.size(1), self.t_states).unsqueeze(3)    
    if self.use_scalar == 0:
      t_emb = self.t_emb
      nt_emb = self.nt_emb
      root_emb = self.root_emb
      root_emb = root_emb.expand(batch_size, self.state_dim)
      t_emb = t_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, n, self.t_states, self.state_dim)
      nt_emb = nt_emb.unsqueeze(0).expand(batch_size, self.nt_states, self.state_dim)
      root_scores = F.log_softmax(self.root_mlp(root_emb), 1)
      unary_scores = F.log_softmax(self.vocab_mlp(t_emb), 3)
      unary = torch.gather(unary_scores, 3, x_expand).squeeze(3)
      rule_score = F.log_softmax(self.rule_mlp(nt_emb), 2) # nt x t**2
      rule_scores = rule_score.view(batch_size, self.nt_states, self.all_states, self.all_states)
    else:
      root_scores = F.log_softmax(self.root_logits, 0).unsqueeze(0).expand(
        batch_size, self.nt_states)
      rule_scores = F.log_softmax(self.rule_logits, 1).unsqueeze(0).expand(
        batch_size, self.nt_states, self.all_states**2).view(
          batch_size, self.nt_states, self.all_states, self.all_states)
      unary_scores = F.log_softmax(self.emission_logits, 1).unsqueeze(0).unsqueeze(0).expand(
        batch_size, n, self.t_states, self.vocab)
      unary = torch.gather(unary_scores, 3, x_expand).squeeze(3)
    log_Z = self.pcfg._inside(unary, rule_scores, root_scores)
    if argmax:
      with torch.no_grad():
        max_score, binary_matrix, spans = self.pcfg._viterbi(unary, rule_scores, root_scores)
        self.tags = self.pcfg.argmax_tags
      return -log_Z, binary_matrix, spans
    else:
      return -log_Z
      
  def get_rule_probs(self):
    if self.use_scalar == 1:
      root_probs = F.softmax(self.root_logits, 0)      
      rule_probs = F.softmax(self.rule_logits, 1)                             
      emission_probs = F.softmax(self.emission_logits, 1) 
    else:
      root_probs = F.softmax(self.root_mlp(self.root_emb).squeeze(0), 0)
      rule_probs = F.softmax(self.rule_mlp(self.nt_emb), 1)
      emission_probs = F.softmax(self.vocab_mlp(self.t_emb), 1)
    return root_probs, rule_probs, emission_probs
                             
