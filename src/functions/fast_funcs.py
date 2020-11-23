import warnings
warnings.filterwarnings('ignore')

import numpy as np
import csv, os, sys
from mpl_toolkits.mplot3d import axes3d
from scipy import spatial
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator
from sklearn import preprocessing
from sklearn import model_selection
import time
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, 'src')
#assuming cloned `https://github.com/paulcon/active_subspaces` into local repos folder
sys.path.insert(0, 'active_subspaces')

from functions.ground.generalFunctions import *
import functions.ground.base_am_fxns as bf

import active_subspaces as ac


# Script for building and testing active manifold algo
#
# Contains modified functions from `functions.ground.base_am_fxns.py` and `functions.error_mains.py`
#  * Optimizes functions for building the manifold and finding closest points by building kd trees only once
#  * bug fix for setting random seeds
#  * bug fix for finding point in [0,1] corresponding to intersection between level set and am in `find_AM_intersection` (last line)
#  * updates code to reflect changes in algorithm to deal with:
#       * v parallel to grad v issue
#       * test paths walking outside hypercube
#       * paths that end far from AM (due to path crossing back on itself)


def make_mesh(dim, step=0.02, numpts=None):
    """
    Create mesh of uniformly sampled pts over [-1,1]^dim
    Input:  dim -- int, dimension of space m
            stepsize -- int or float, distance between mesh points
                (default = .02) (ignored if numpts is passed)
            numpts -- number of points to sample along 1 axis

    Output:    meshPoints -- list, points of mesh arranged lexicographically
    """
    if numpts is None:
    # Define mesh
        numpts = (2. // step) + 1.
    mesh = np.meshgrid(*[np.linspace( -1., 1., numpts) for i in xrange(dim)])

    # Make mesh into list of points
    # Order them lexicographically
    mesh = [mesh[i].reshape((np.prod(mesh[i].shape),)) for i in np.arange(len(mesh))]
    meshPoints = sorted(zip(*mesh))

    return meshPoints

def build_data(dim, numPoints, f, grad_f):
    """
    build 'numPoints' test points of uniformly distributed across [-1,1]^dim
    and return points X, f(X), and gradiant paths(X)
    """
    meshy = make_mesh(dim = dim,numpts=numPoints)
    f_x = bf.sample_f_on_mesh(f,meshy)
    amGrads = bf.compute_paths(grad_f,meshy)[0]
    return meshy, f_x, amGrads

def get_random_init_pt(dim, seed = 0):
    """
    Get random point in mesh in [-1,1]^dim using seed 'seed'

    Inputs:  dim -- int, dimension of hypercube
             seed -- int, seed for random initial point (default = 0)
    Output: 1d np.array of len == dim
    """
    np.random.seed(seed)
    return np.ravel(2*np.random.rand(dim,1)-1)

def build_AM_from_data(seedPoint,mesh_kdtree, fSamples, gradPaths, stepsize):
    """
    Build active manifold thru seedPoint on data held in mesh_kdtree

    Input:  seedPoint -- array or list, random REGULAR value in [-1,1]^m (no CP)
            mesh_kdtree -- sklearn.spatial.KDTree of meshPoints
            fSamples -- array, f evaluated at meshPoints
            gradPaths -- array, grad f evaluated at meshPoints
            stepsize -- float, specifies how large to step

    Output: activeManifold -- array, ordered points of the active manifold
            fValues -- array,
    """

    # Define region / hypercube [-1,1]^(m+1)
    dim = len(seedPoint)
    rBound = np.ones(dim)
    #rBound = np.append(np.ones(liftDim - 1),t_0)
    Dom = spatial.Rectangle( -1*rBound, rBound )


    # Initialize activeManifold and fValues lists
    p0 = seedPoint

    # Find index of closest mesh point to seedpoint
    # Use d0 for first direction
    i0 = mesh_kdtree.query(p0)[1]
    d0 = gradPaths[i0]

    # Initialize gradient ascent
    ascentPoints = np.asarray(p0)
    aCloseVals = np.asarray(fSamples[i0])

    # Take one step
    p = p0 + (stepsize * d0)

    ascentPoints = np.vstack((ascentPoints,p))

    i1 = mesh_kdtree.query(p)[1]
    aCloseVals = np.append(aCloseVals, fSamples[i1])

    cond = np.array(1)
    # Gradient Ascent
    while Dom.min_distance_point(ascentPoints[-1]) == 0 and min(cond.flatten()) > stepsize/3:

        i = mesh_kdtree.query(ascentPoints[-1])[1]
        d = gradPaths[i]

        p = ascentPoints[-1] + (stepsize * d)
        ascentPoints = np.vstack((ascentPoints, p))

        i1 = mesh_kdtree.query(p)[1]
        aCloseVals = np.append(aCloseVals, fSamples[i1])

        #update loop condition
        cond = spatial.distance.cdist([ascentPoints[-1]], ascentPoints[0:len(ascentPoints)-1], 'euclidean')

    # Delete last elements (outside of hypercube)
    ascentPoints = np.delete(ascentPoints, len(ascentPoints) - 1, 0)
    aCloseVals = np.delete(aCloseVals, len(aCloseVals) - 1, 0)

    # Initialize gradient descent
    descentPoints = np.asarray(p0)
    dCloseVals = fSamples[i0]

    # Take one step
    p = p0 - (stepsize)*(d0)

    descentPoints = np.vstack((descentPoints,p))

    i1 = mesh_kdtree.query(p)[1]
    dCloseVals = np.append(dCloseVals, fSamples[i1])

    cond = np.array(1)
    # Gradient Descent
    while Dom.min_distance_point(descentPoints[-1]) == 0 and min(cond.flatten()) > stepsize/3:

        i = mesh_kdtree.query(descentPoints[-1])[1]
        d = gradPaths[i]

        p = descentPoints[-1] - (stepsize * d)
        descentPoints = np.vstack((descentPoints,p))

        i1 = mesh_kdtree.query(p)[1]
        dCloseVals = np.append(dCloseVals, fSamples[i1])

        #update loop condition
        cond = spatial.distance.cdist([descentPoints[-1]], descentPoints[0:len(descentPoints)-1], 'euclidean')

    # Delete first and last element in descentpoints and fValuesdescent
    descentPoints = np.delete(descentPoints, 0, 0)
    descentPoints = np.delete(descentPoints, len(descentPoints) - 1, 0)
    dCloseVals = np.delete(dCloseVals, 0)
    dCloseVals = np.delete(dCloseVals, len(dCloseVals) - 1)

    # Flip order of descentPoints and concatenate lists
    activeManifold = np.concatenate((np.flipud(descentPoints), ascentPoints), axis=0)
    fCloseVals = np.concatenate((np.flipud(dCloseVals), aCloseVals))

    return activeManifold, fCloseVals

def project_to_AM(startPoint,mesh_kdtree, am_kdtree, activeManifold, gradPaths, stepsize, debug = False):
    """
    Project startPoint to active manifold:
        find path along level set to within stepsize of AM
        if algo stops due to another stopping condition,
         return a path with a null pt (np.nan array) to end of path

    Input:  startPoint -- array or list, point in [-1,1]^m
            mesh_kdtree -- sklearn.spatial.KDTree of meshPoints
            am_kdtree -- sklearn.spatial.KDTree of pts in active manifold
            activeManifold -- array, ordered points of the active manifold in R^m
            gradPaths -- array, grad f evaluated at meshPoints
            stepsize -- float, size of steps for the algorithm
            debug -- boolean, if True (default = False), return everything computed during algo run

    Output:    levSet -- array, points of level set with levSet[0] = startPoint, arranged vertically
    """

    # Initialize quantities of interest
    p0 = list(startPoint)
    levSet = list()

    levSet.append(p0)

    # Find index of closest point on active manifold to starting point
    manDist0, manInd0 = am_kdtree.query(p0)
    m0 = activeManifold[manInd0]

    # Check to see if starting point is already on the active manifold
    if manDist0 <= stepsize:
        closeIndex = manInd0
        if debug: print 'Point is nearly active manifold point %s' %(m0)
        levSet.append(list(m0))

    if np.any(np.abs(p0)>1):
        levSet.append(list([np.nan]*len(startPoint)))
        return np.vstack(levSet)

    else:

        # Make vector m0 - p0

        v0 = normalize(m0 - p0)

        # Find index of closest mesh point to p0 and its basePath
        meshDist0, meshInd0 = mesh_kdtree.query(p0)

        n0 = gradPaths[meshInd0]


        # Take a step along [x]
        if np.abs(np.dot(n0, v0)) > .95:
            if debug:  print 'towards origin'
            v0 = normalize(np.zeros_like(n0) - p0)
            if np.abs(np.dot(n0, v0)) > .95:
                if debug:  print('orth')
                v0 = get_orthonormal_vector(n0)
            #print("find perp dir")

        d0 = normalize(v0 - np.dot(v0, n0)*n0)

        #d0 = w0 / np.linalg.norm(w0)
        p = p0 + (stepsize)*d0

        levSet.append(list(p))


        # Initialize these for the loop
        manDist = manDist0
        manInd = manInd0

        # Initialize loop condition; stepsize/n chosen to taste
        cond = np.array(1)

        # While loop checks distance of levSet[-1] from manifold AND from the rest of levSet
        # Tolerance factor stepsize/3 adjustable to taste
        while min(cond.flatten()) > stepsize/3 and manDist >= stepsize:

            if np.any(np.abs(p)>1):
                levSet.append(list([np.nan]*len(startPoint)))
                return np.vstack(levSet)

            #print "level set point %s" %(levSet[-1])
            closeIndex = manInd

            manDist, manInd = am_kdtree.query(levSet[-1])
            m = activeManifold[manInd]

            v = normalize(m - levSet[-1])

            meshDist, meshInd = mesh_kdtree.query(levSet[-1])
            n = gradPaths[meshInd]

            if np.abs(np.dot(n, v)) > .95:
                v = normalize(np.array(levSet[-1]) - np.array(levSet[-2]))
                if np.abs(np.dot(n, v)) > .95:
                    v = normalize(np.zeros_like(n) - levSet[-1])
                    if np.abs(np.dot(n, v)) > .95:
                        v = get_orthonormal_vector(n)

            d = normalize(v - np.dot(v, n)*n)

            p = levSet[-1] + (stepsize * d)

            levSet.append(list(p))

            #update loop condition
            cond = spatial.distance.cdist([levSet[-1]], levSet[0:len(levSet)-1], 'euclidean')

    if manDist > stepsize: levSet.append(list([np.nan]*len(startPoint)))
    return np.vstack(levSet)

def get_orthonormal_vector(n0):
    """
    Get normalized vector orthogonal to 'n0'
    """
    nzero_idx = np.where(n0 != 0)[0][0]
    v0 = np.zeros_like(n0)
    other_idx = 0
    if nzero_idx == 0:
        other_idx = 1
    v0[nzero_idx] = -n0[other_idx]
    v0[other_idx] = n0[nzero_idx]
    assert abs(np.dot(v0, n0)) < .001 #make sure it is orthogonal
    return normalize(v0)

def normalize(x):
    """
    Normalize vector x
    """
    return x/math.sqrt(np.sum(x**2))

def find_AM_intersection(levSet, activeManifold, mesh_kdtree, am_kdtree, meshPoints, gradPaths):
    """
    FInd intersection of level set path with active manifold

    Input:  levSet -- array, points of level set with levSet[0] = startPoint
            activeManifold -- array, ordered points of the active manifold
            mesh_kdtree -- sklearn.spatial.KDTree of meshPoints
            am_kdtree -- sklearn.spatial.KDTree of pts in active manifold
            meshPoints -- array, points of mesh arranged vertically
            gradPaths -- array, gradient evaluated at mesh points, normalized, arranged vertically

    Output: q -- float, approximate point of level set and active manifold intersection
            s -- float in [0,1], parameter corresponding to point q on the p.w.-linear interpolation
                to am curve, M(s).
    """
    # define closest point (stopping point) to active manifold
    # Need condition for only 1 point in array

    if levSet.shape == (len(meshPoints[0]),):
        p = levSet

    else:
        p = levSet[-1]

    # Find index of closest base point to p; use it to define path
    pIndex = mesh_kdtree.query(p)[1]
    n = gradPaths[pIndex]

    # Find closest two points on active manifold to p
    _, indices = am_kdtree.query(p, k=2)
    m1, m2 = activeManifold[indices]

    # Solve for t in line interpolating m1 and m2, noting that we want <v + tw, n> = 0
    v = m1 - p
    w = m2 - m1

    # Include condition for <w, n> = 0
    if np.dot(w, n) < 0.001:
        t = 0

    else:
        t = np.dot(-v, n) / np.dot(w, n)

    # Compute q as convex combination of m1 and m2
    q = t * m2 + (1 - t) * m1

    # Define map from a.m. to [0,1]
    numpts = len(activeManifold)
    sValues = np.linspace(0., numpts, numpts) / (numpts)

    # Compute s: the image of q under this map
    s = t * sValues[indices[1]] + (1 - t) * sValues[indices[0]]

    return q, s


#Main AM functions
def mainRandEx(inputs, fSamples, grads,stepsize = None,
    test_train_split_seed = 0, init_pt_seed = 0, seedPoint = None,
     test_size = .2, debug = False, error_threshold = None):
    """
    Run active manifold algorithm: build AM thru seedPoint on (1-test_size) of the input data,
    and compute error between f_vals of test pts and f_value of test pt projected onto AM

    Input:  meshPoints -- array, points of mesh arranged vertically
            fSamples -- array, f evaluated at meshPoints
            grads -- array, gradient evaluated at mesh points arranged vertically
            test_size -- float in [0,1], fraction of test points in test/train split
            stepsize -- float, specifies how large to step (default= 2/3*(1/numPoints)*sqrt(dim))
            test_train_split_seed -- int, random seed for splitting into test/train set (default = 0)
            init_pt_seed -- int, seed for random initial point (default = 0) (ignored if SeedPoint is passed)
            seedPoint -- np.arr of len == dim, random REGULAR value in [-1,1]^dim (no CP)
                (pass none if using init_pt_seed for random start pt)
            outpath -- filepath, csv filepath to write results to (default = None)
            return_all -- if True (default = False)
                return dictionary of  everything computed during algo run, and results
                (for debugging or plotting, see if debug == True in fast_funcs.mainRandEx() for details on these vars)

    Output: fitErrorL1 (float), fitErrorL2 (float), n_testpts_used (int)*
        *number of test points where algo successfully found path that intersected with manifold-- these are pts used in calculating error
    """
    #get random initial point
    dim = len(inputs[0])
    if seedPoint is None:
        seedPoint = get_random_init_pt(seed = init_pt_seed, dim = dim)

    if stepsize is None:
        stepsize = 2./3*(1./int(math.sqrt(len(inputs))))*math.sqrt(dim)

    #Generate train/test data using sklearn (Should switch over to this version once no longer running tests comparing anthonys to mikis code)
    X_train, X_test, y_train, y_test, grad_train, grad_test= model_selection.train_test_split(
        inputs, fSamples, grads, test_size = test_size, random_state = test_train_split_seed)

    # Compute gradients at trainingPoints
    gradPaths = preprocessing.normalize(grad_train)

    # Build KD Tree for mesh
    mesh_kdtree = spatial.KDTree(X_train)

    # Build active manifold from training points
    activeManifold, fCloseVals = build_AM_from_data(
        seedPoint = seedPoint, mesh_kdtree = mesh_kdtree,
        fSamples= y_train, gradPaths=gradPaths, stepsize=stepsize)

    # Build KD Tree active manifold
    am_kdtree = spatial.KDTree(activeManifold)

    # Fit to pw-cubic Hermite interpolant
    numpts = len(activeManifold)
    sValues = np.linspace(0., numpts, numpts) / (numpts)
    splinef = PchipInterpolator(sValues, fCloseVals)


    proj_pts = multi_arg_map(project_to_AM, X_test, mesh_kdtree, am_kdtree, activeManifold, gradPaths, stepsize)

    # Compute q,s values for each random sample, organize into list
    no_intersection_msk =  np.isnan(np.array([path[-1] for path in proj_pts ])).any(1)

    paths = [path for i, path in enumerate(proj_pts) if not no_intersection_msk[i]]
    non_intersecting_paths = [path[:-1] for i, path in enumerate(proj_pts) if no_intersection_msk[i]]

    n_testpts_used = len(paths)
    intersection_tup = multi_arg_map(find_AM_intersection, paths, activeManifold, mesh_kdtree, am_kdtree, X_train, gradPaths)

    # Compute approximations at appropriate points
    fFitApprox = np.hstack([splinef(tup[1]) for tup in intersection_tup])

    # Average L1, L2 error of the fit
    fitErrorL1 = np.mean(np.abs(y_test[~no_intersection_msk] - fFitApprox))
    fitErrorL2 = np.linalg.norm(fFitApprox - y_test[~no_intersection_msk]) / float(len(X_test))
    # print 'The L1 Error is %f' %fitErrorL1
    # print 'The L2 Error is %f' %fitErrorL2

    if debug:
        ptwise_error = np.abs(y_test[~no_intersection_msk] - fFitApprox)

        return {"X_train" :X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test" : y_test,
                "grad_train" : grad_train,
                "grad_test" : grad_test,
                "gradPaths" : gradPaths,
                "activeManifold" : activeManifold,
                "mesh_kdtree": mesh_kdtree,
                "am_kdtree" : am_kdtree,
                "fCloseVals" : fCloseVals,
                "paths": paths,
                "non_intersecting_paths": non_intersecting_paths,
                "no_intersection_msk": no_intersection_msk,
                "intersection_tup": intersection_tup,
                "ptwise_error" : ptwise_error,
                "fitErrorL1" : fitErrorL1,
                "fitErrorL2" : fitErrorL2,
                "n_testpts_used" : n_testpts_used}

    return fitErrorL1, fitErrorL2, n_testpts_used

# def _get_good_bad_path_idxs(ptwise_error, error_threshold = 10, verbose = False):
#     bad_idxs = np.where(ptwise_error>=error_threshold)[0]
#     good_idxs = np.where(ptwise_error<error_threshold)[0]
#     if verbose: print "number of bad/good test pts: %d/%d" %(len(bad_idxs), len(good_idxs))
#     return good_idxs, bad_idxs


#NOT USED (just for testing)
def mainRandEx_old(inputs, fSamples, grads, stepsize, test_size, seedPoint,
    seed = 0, outpath = None, verbose = False):
    """
    Anthonys function for testing error between function and active manifold
    """
    # Random Samples to Test function approximation on

    meshPoints, testPoints,fSampsTraining, fSampsTest, meshgrads, testgrads = model_selection.train_test_split(
        inputs, fSamples, grads, test_size = test_size, random_state = seed)

    # Compute gradient
    gradPaths = preprocessing.normalize(meshgrads)

    # Build active manifold
    activeManifold, fCloseVals = bf.build_AM_from_data(seedPoint, meshPoints, fSampsTraining, gradPaths, stepsize)

    # Fit to model function
    numpts = len(activeManifold)
    sValues = np.linspace(0., numpts, numpts) / (numpts)
    splinef = PchipInterpolator(sValues, fCloseVals)
    # Initialize levSetList, closeIndexList
    levSetList = []
    closeIndexList = []

    # Compute points of [x] and indices of closest pt on a.m. to [x], for each x
    for i in xrange(len(testPoints)):
        if verbose:
            print "level set for data point  %s,  %s of %s" %(testPoints[i], i+1, len(testPoints))
        levSet, closeIndex = bf.project_to_AM(testPoints[i], meshPoints, activeManifold, gradPaths, stepsize)
        levSetList.append(levSet)
        closeIndexList.append(closeIndex)

    #print(levSetList[5])
    # Initialize qList, sList
    qList = []
    sList = []

    # Compute q,s values for each random sample, organize into list
    for i in xrange(len(levSetList)):
        q, s = bf.find_AM_intersection(levSetList[i], activeManifold, closeIndexList[i], meshPoints, gradPaths)
        qList.append(q)
        sList.append(s)

    # Compute approximations at appropriate points
    s = sList[0]
    fFitApprox = splinef(s)
    for i in range(1, len(sList)):
        s = sList[i]
        fFitApprox = np.append(fFitApprox, splinef(s))

    # Average L1, L2 error of the fit
    fitErrorL1 = np.mean(np.abs(fSampsTest - fFitApprox))
    fitErrorL2 = np.linalg.norm(fFitApprox - fSampsTest) / float(len(testPoints))

    results = {'fit L1 error': fitErrorL1, "fit L2 error": fitErrorL2}
    if outpath:
        jsonify(results, outpath)

    print 'The L1 Error is %f' %fitErrorL1
    print 'The L2 Error is %f' %fitErrorL2

    return fitErrorL1, fitErrorL2

#Main AS function
def SubspEx(inputs, fSamples, grads, test_train_split_seed = 0,test_size = .2, outpath=None):
    """
    Function for building active subspace using Constantines method
    """
    # Generate training/testing points based on partNum
    X_train, X_test, y_train, y_test, grad_train, grad_test= model_selection.train_test_split(
        inputs, fSamples, grads, test_size = test_size, random_state = test_train_split_seed)

    # Build active subspace from trainingPoints with 100 bootstrap replicates
    ss = ac.subspaces.Subspaces()
    ss.compute(df=grad_train, nboot=100)
    ss.partition(1)
    y = np.dot(X_train,ss.W1)

    # Build polynomial approximation to the data, using Constantine functions
    RS = ac.subspaces.PolynomialApproximation(2)
    y_train = y_train.reshape((len(y_train),1))
    RS.train(y, y_train)
    #print 'The R^2 value of the response surface is {:.4f}'.format(RS.Rsqr)

    # Plot it
    # plt.figure(figsize=(7, 7))
    # y0 = np.linspace(-1, 1, 200)
    # plt.scatter(y, y_train, c = '#66c2a5')
    # plt.plot(y0, RS.predict(y0[:,None])[0], c = '#fc8d62', ls='-',linewidth=2)
    # plt.grid(True)
    # plt.xlabel('Active Variable Value', fontsize=18)
    # plt.ylabel('Output', fontsize=18)

    #avdom = domains.BoundedActiveVariableDomain(ss)
    #avmap = domains.ActiveVariableMap(avdom)

    #Project testPoints to the AS and compute approximate fVals
    asProjPts = np.dot(X_test,ss.W1)
    fApproxVals = RS.predict(asProjPts)[0]

    # Average L1, L2 errors
    fitErrorL1 = np.mean(np.abs(y_test - fApproxVals))
    fitErrorL2 = np.linalg.norm(fApproxVals - y_test) / float(len(X_test))

    results = {'fit L1 error': fitErrorL1, "fit L2 error": fitErrorL2}
    if outpath:
        jsonify(results, outpath)
    #
    # print 'The L1 Error is %f' %fitErrorL1
    # print 'The L2 Error is %f' %fitErrorL2

    return fitErrorL1, fitErrorL2

#general function
def multi_arg_map(f, data, *args):
    """
    map function with multiple arguments
    """
    return map(lambda x: f(x,*args), data)
