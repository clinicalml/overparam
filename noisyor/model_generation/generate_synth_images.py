import numpy as np
import sys
import os

''' Generates a noisy-OR model directory with the model of the IMG dataset. '''


images = []

   ##
  #  #
 #    #
#      #
#      #
 #    #
  #  #
   ##
im = np.zeros((8,8))
im[3:5, 0] = 1
im[3:5, -1] = 1
im[0, 3:5] = 1
im[-1, 3:5] = 1
im[1, [2,5]] = 1
im[2, [1,6]] = 1
im[-2, [2,5]] = 1
im[-3, [1,6]] = 1
images.append(im)

    #
    #
    #
    ######
######
     #
     #
     #
im = np.zeros((8,8))
im[4, :5] = 1
im[:5, 3] = 1
im[3, 4:] = 1
im[4:, 4] = 1
images.append(im)

########
#      #
#      #
#      #
#      #
#      #
#      #
#      #
########
im = np.zeros((8,8))
im[0,:] = 1
im[:,0] = 1
im[-1,:]= 1
im[:,-1]= 1
images.append(im)

#
 #
  #
   #
    #
     #
      #
       #
im = np.zeros((8,8))
for i in xrange(8):
    im[i,i] = 1
images.append(im)

       #
      #
     #
    #
   #
  #
 #
#
im = np.zeros((8,8))
for i in xrange(8):
    im[i,8-1-i] = 1
images.append(im)

 ######
 #    #
 #    #
 #    #
 #    #
 ######

im = np.zeros((8,8))
im[1,1:-1] = 1
im[1:-1,1] = 1
im[-2,1:-1]= 1
im[1:-1,-2]= 1
images.append(im)

  ####
  #  #
  #  #
  ####

im = np.zeros((8,8))
im[2,2:-2] = 1
im[2:-2,2] = 1
im[-3,2:-2]= 1
im[2:-2,-3]= 1
images.append(im)





 ###
 ###
 ###

im = np.zeros((8,8))
im[4:7, 1] = 1
im[4:7, 2] = 1
im[4:7, 3] = 1
images.append(im)


if len(sys.argv) < 2:
    print "usage: python generate_synth_images.py model_path"
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
    print >>prior, 0.25
    for pixel in xrange(64):
        print >>weights, '\t', 1-images[im].flatten()[pixel]*0.9,
    print >>weights, ""

print >>noise, 64
for pixel in xrange(64):
    print >>noise, 0.999

prior.close()
weights.close()
noise.close()
