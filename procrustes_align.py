# -*- coding: utf-8 -*-
""" Generates the outputs of an arbitrary CNN layer. """

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"
__date__ = "14 september 2016"

import os
import numpy as np
import pylab
import cv2
import errno
import os

DEST='Aligned'
CROP=75 # from -75 to 75 = 125px
DEBUG=False
RESIZE=256 # final size

#  http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# http://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy
def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform


def load_data(root='AAM_landmarks', force=False):
    if (not force) and os.path.exists('landmarks.npy'):
        data = np.load('landmarks.npy').item()
        landmarks = data['landmarks']
        images = data['images']
        return landmarks, images, data['subjects'], data['sequences']
    else:
        subject = []
        sequence = []
        counter = 0
        dirlist = os.listdir(root)
        images = []
        data = []
        seq_num = []
        curr_seq = 0
        # TODO make it completely recursive
        for d in dirlist:
            path = os.path.join(root, d)
            for d2 in os.listdir(path):
                path2 = os.path.join(path, d2)
                for im in os.listdir(path2):
                    path3 = os.path.join(path2, im)
                    if 'txt' in path3:
                        with open(path3) as infile:
                            im_path = path3.replace(root, 'Images').replace('_aam.txt', '.png')
                            image = cv2.imread(im_path)
                            if image.mean() < 10:
                                print 'skip'
                                continue
                            images.append(im_path)
                            r = np.genfromtxt(infile)
                            #r /= imsize
                            #r -= r.mean(axis=0)
                            data.append(r)
                            subject.append(d)
                            if len(seq_num) > 0 and sequence[-1] != d2:
                                curr_seq += 1
                            seq_num.append(curr_seq)
                            sequence.append(d2)

            counter += 1
            print counter, ' of ', len(dirlist)
        data = np.array(data)
        np.save('landmarks.npy', {'landmarks': data, 'images':images, 'subjects':subject, 'sequences':sequence, 'seq_num':seq_num})
        return data, images, subject, sequence

def generalized_procrustes(data):
    mean = data[0,...]
    print 'Aligning'
    d = 100
    d_old = 100
    while d > 0.0001:
        d_new = 0
        for i in xrange(data.shape[0]):
            d_, data[i,:], _ = procrustes(mean, data[i,:], scaling=False, reflection=False)
            d_new += d_ / data.shape[0]
        d = d_old - d_new
        d_old = d_new
        mean = data.mean(axis=0)
    return mean


# 1. load data from directory structure
data, images, subjects, sequences = load_data(force=False)
data_copy = data.copy()

# 2. Align landmarks
mean = generalized_procrustes(data)

# 3. Align images and crop
print 'Processing data'
for index in xrange(len(images)):
    landmark = data_copy[index,:]
    image = cv2.imread(images[index])
    _, Z, tform = procrustes(mean, landmark)

    rotate = np.identity(3)
    rotate[:2,:2] = tform['rotation'].transpose()
    to_centroid = np.identity(3)
    to_centroid[:2,-1] = landmark.mean(axis=0)
    to_orig = np.identity(3)
    to_orig[:2, -1] = -landmark.mean(axis=0)
    translate = np.identity(3)
    translate[:2,-1] = tform['translation']

    scale = np.identity(3)
    scale[0,0] = tform['scale']
    scale[1,1] = tform['scale']

    M_translate = np.dot(translate, scale)
    M = np.dot(M_translate, rotate)

    rows, cols, _ = image.shape
    im2 = cv2.warpAffine(image, M[:-1,:], (cols,rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    if DEBUG: # compute transformed landmarks manually
        Z = landmark.copy()
        Z = np.dot(Z, tform['rotation'])
        Z *= tform['scale']
        Z += tform['translation']
        Z = np.floor(Z).astype('int')

    zentroid = Z.mean(axis=0)
    min_x = max(0,zentroid[0] - CROP)
    max_x = min(zentroid[0] + CROP, im2.shape[1]-1)
    min_y = max(0, zentroid[1] - CROP)
    max_y = min(zentroid[1] + CROP, im2.shape[0]-1)
    im2 = im2[min_y:max_y,min_x:max_x,:]
    dest_path = os.path.join(DEST, subjects[index], sequences[index])
    mkdir_p(dest_path)
    cv2.imwrite(os.path.join(dest_path, os.path.basename(images[index])), cv2.resize(im2, (RESIZE,RESIZE), interpolation=cv2.INTER_CUBIC))
    if index % 100 == 0:
        print os.path.join(dest_path, os.path.basename(images[index]))
        print index, '/', len(images)

    if DEBUG: # show landmarks
        import pylab
        for i in xrange(mean.shape[0]):
            im2[(mean[i,1]-1):(mean[i,1]+1),(mean[i,0]-1):(mean[i,0]+1), :] = [255,0,0]
            im2[(Z[i, 1] - 1):(Z[i, 1] + 1), (Z[i, 0] - 1):(Z[i, 0] + 1), :] = [0, 255, 0]
            im2[(landmark[i, 1] - 1):(landmark[i, 1] + 1), (landmark[i, 0] - 1):(landmark[i, 0] + 1), :] = [0, 0, 255]

        pylab.subplot(1,2,1)
        pylab.imshow(im2)
        pylab.subplot(1,2,2)
        pylab.imshow(image[:,:,::-1])
        pylab.waitforbuttonpress()
        pylab.close('all')
