import numpy as np
import sys
import os

''' Generates a noisy-OR model directory with the model of the CON8 or CON24 datasets. '''


images = []

np.random.seed(1)
L = 24 # Set to 8 for CON8 or 24 for CON24.
for source in range(8):
    im = np.zeros(64)
    im[np.random.choice(64, L, replace=False)] = 1
    images.append(im.reshape((8, 8)))


if len(sys.argv) < 2:
    print "usage: python generate_synth_images_adversarial.py model_path"
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
    print >>prior, 1.0 / 8
    for pixel in xrange(64):
        print >>weights, '\t', 1-images[im].flatten()[pixel]*0.9,
    print >>weights, ""

print >>noise, 64
for pixel in xrange(64):
    print >>noise, 0.999

prior.close()
weights.close()
noise.close()
