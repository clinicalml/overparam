import numpy as np
import sys

import utils


''' Learns a sparse coding model using the alternating minimization algorithm
    described in Li, Liang & Risteki (2016).
    (See https://papers.nips.cc/paper/6417-recovery-guarantee-of-non-negative-matrix-factorization-via-alternating-updates.pdf) '''


config = {
    'seed': None,
    'step_size': 0.1,
    'batch_size': 20,
    'relu_threshold': 0.005,
    'save_every': 100,
    'epochs': 2000,
    'heldout': 100
}
batch_size = config['batch_size']

if len(sys.argv) < 4:
    print "usage: python train_relu.py samples_path output_path topics seed"
    sys.exit()
samples_path = sys.argv[1]
output_path = sys.argv[2]
K = int(sys.argv[3])
if len(sys.argv) > 4:
    config['seed'] = int(sys.argv[4])
    np.random.seed(config['seed'])

samples = utils.read_samples(samples_path)
N = samples.shape[1]

samples_heldout = np.copy(samples[: config['heldout']])
samples = np.copy(samples[config['heldout'] :])
np.random.shuffle(samples)
S = samples.shape[0] # number of samples
S_heldout = samples_heldout.shape[0] # number of heldout samples

A = utils.get_initial_A_gaussian(N, K)

sys.stdout = open(output_path + '/' + 'log.txt', 'w')
print config

# Train.
for epoch in range(config['epochs']):
    print 'Epoch', epoch
    sys.stdout.flush()

    # Save model parameters.
    if (epoch + 1) % config['save_every'] == 0:
        outfile = open(output_path + '/' + 'model_epoch' + str(epoch + 1) + '.dat', 'w')
        outfile.write(str(N) + ' ' + str(K) + '\n')
        for i in range(N):
            for j in range(K):
                outfile.write(str(A[i, j]) + ' ')
            outfile.write('\n')
        outfile.close()

    shuffle = np.random.permutation(S)
    batch = np.zeros((batch_size, N))

    reconstruction_error = 0.0

    # Optimize.
    for i in range(0, S, batch_size):
        for j in range(i, min(i + batch_size, S)):
            batch[j - i] = samples[shuffle[j]]

        pseudoinverse = np.linalg.pinv(A)
        current_x = np.matmul(batch, np.transpose(pseudoinverse))
        current_x = np.vectorize(utils.get_ReLU(config['relu_threshold']))(current_x)
        reconstruction_error += np.linalg.norm(np.matmul(current_x, np.transpose(A)) - batch) ** 2

        gradient_A = np.matmul(np.transpose(np.matmul(current_x, np.transpose(A)) - batch), current_x)
        gradient_A /= batch_size

        A -= config['step_size'] * gradient_A
        for i in range(K):
            A[:, i] /= np.linalg.norm(A[:, i])
    reconstruction_error /= S

    # Compute heldout reconstruction error.
    reconstruction_erorr_heldout = 0.0
    if S_heldout != 0:
        pseudoinverse = np.linalg.pinv(A)
        current_x = np.matmul(samples_heldout, np.transpose(pseudoinverse))
        current_x = np.vectorize(utils.get_ReLU(config['relu_threshold']))(current_x)
        reconstruction_erorr_heldout = np.linalg.norm(np.matmul(current_x, np.transpose(A)) - samples_heldout) ** 2
        reconstruction_erorr_heldout /= S_heldout

    print 'Reconstruction error (train, heldout)', reconstruction_error, reconstruction_erorr_heldout
