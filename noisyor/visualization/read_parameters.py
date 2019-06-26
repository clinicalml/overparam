import os
import sys
import numpy as np

def read_true_parameters(model_path):
    p_file = file(model_path + '/priors.txt')
    f_file = file(model_path + '/weights.txt')
    n_file = file(model_path + '/noise.txt')

    dim_latent = int(p_file.readline())
    p = np.zeros(dim_latent)
    for i in range(dim_latent):
        p[i] = float(p_file.readline())

    dim_observed = int(n_file.readline())
    n = np.zeros(dim_observed)
    for i in range(dim_observed):
        n[i] = float(n_file.readline())

    f = np.zeros((dim_latent, dim_observed))
    for i in range(dim_latent):
        line = f_file.readline()
        f[i] = np.array([float(x) for x in line.split()])

    return p, f, n

def read_parameters(model_path):
    infile = file(model_path)

    line = infile.readline()
    dim_latent, dim_observed = [int(x) for x in line.split()]

    line = infile.readline()
    p = np.array([float(x) for x in line.split()])
    f = np.zeros((dim_latent, dim_observed))
    for i in range(dim_latent):
        line = infile.readline()
        f[i] = np.array([float(x) for x in line.split()])
    line = infile.readline()
    n = np.array([float(x) for x in line.split()])

    return p, f, n

def read_recognition_network(model_path):
    infile = file(model_path)

    line = infile.readline()
    dim_latent, dim_observed = [int(x) for x in line.split()]

    w = np.zeros((dim_observed, dim_latent))
    for i in range(dim_observed):
        line = infile.readline()
        w[i] = np.array([float(x) for x in line.split()])
    line = infile.readline()
    b = np.array([float(x) for x in line.split()])

    return w, b

def read_likelihood(log_path):
    infile = file(log_path)
    lines = infile.readlines()
    return float(lines[-1].strip().split()[-1])

def read_likelihood_epoch(log_path, epoch):
    infile = file(log_path)
    lines = infile.readlines()

    okay = False
    for i in range(len(lines)):
        if lines[i].startswith('Epoch ' + str(epoch)):
            okay = True
        if okay and lines[i].startswith('Log likelihood'):
            return float(lines[i].strip().split()[-1])

    return 0.0

def read_best_likelihood(log_path, frequency):
    infile = file(log_path)
    lines = infile.readlines()

    best_likelihood = np.NINF
    index = 0
    best_index = 0
    for i in range(len(lines)):
        if lines[i].startswith('Log likelihood'):
            index += 1
            if (index + 1) % frequency == 0 and float(lines[i].strip().split()[-1]) > best_likelihood: # i.e. 24-th should be okay with frequency 25
                best_likelihood = float(lines[i].strip().split()[-1])
                best_index = index

    return best_likelihood, best_index
