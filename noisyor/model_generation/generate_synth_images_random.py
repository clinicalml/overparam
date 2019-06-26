import numpy as np
import sys
import os

''' Generates a noisy-OR model directory with the model of the UNIF dataset. '''


images = []

np.random.seed(1)
for source in range(8):
    images.append(np.random.binomial(1, 0.25, size=(16,16)))


if len(sys.argv) < 2:
    print "usage: python generate_synth_images_random.py model_path"
    sys.exit()

model = sys.argv[1]
print "generating network description for", model

print "setting up directories"
os.makedirs(str(model)+"/true_params")

print "writing network parameters"
prior = file(str(model)+'/true_params/priors.txt', 'w')
weights = file(str(model)+'/true_params/weights.txt', 'w')
noise = file(str(model)+'/true_params/noise.txt', 'w')

print >>prior, 8
for im in xrange(8):
    print >>prior, np.random.uniform(0.1, 0.3)
    for pixel in xrange(64):
        print >>weights, '\t', 1-images[im].flatten()[pixel]*np.random.uniform(0.80, 0.95),
    print >>weights, ""

print >>noise, 64
for pixel in xrange(64):
    print >>noise, 0.999

prior.close()
weights.close()
noise.close()
