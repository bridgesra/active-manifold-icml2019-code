# This file contains the basic functions used to run Active Manifolds without lift.

import warnings
warnings.filterwarnings('ignore')

# Imports - can probably be trimmed down
import numpy as np
import matplotlib.pyplot as plt
import csv, os, sys
from mpl_toolkits.mplot3d import axes3d
from scipy import spatial
from scipy.optimize import curve_fit
from sklearn import preprocessing
import random
from generalFunctions import *


# Pointing to the right spot
sys.path.append(os.getcwd() + "/active-manifolds")


# Test Functions
def f1(x,y): return np.exp(y - x**2)

def f2(x,y): return x**2 + y**2

def f3(x,y): return x**3 + y**3 + 0.2*x + 0.6*y

def Squaresum(*vars):
	total = 0
	for i in range(0,len(vars)):
		total += (vars[i])**2
	return total

def cubey(x,y): return y/x**3

def loopy(x,y): return y**2/(1-x**2)


# Gradient Vectors of Test Functions
def gradf1(x,y): return [np.exp(y - x**2)*(-2.)*x, np.exp(y - x**2)]

def gradf2(x,y): return [2.*x,2.*y]

def gradf3(x,y): return [0.2 + (3.)*(x**2), 0.6 + (3.)*(y**2)]

def gradSquaresum(*vars):
	grs = []
	for i in xrange(len(vars)):
		grs =np.concatenate( (grs, np.array([2*vars[i]])), axis = 0)
	return grs

def gcubey(x,y): return [-3*y/x**4, 1/x**3]

def gloopy(x,y): return [2*x*y**2/(1-x**2)**2,2*y/(1-x**2)]


# Model fitting functions (somewhat outdated now - not necessary for the latest version)
def fourpolytest(x,a,b,c,d,e): return a*x**4+b*x**3+c*x**2+d*x+e
def exptest(x,a,b): return a*np.exp(b*x)


# misc functions
def sample_f_on_mesh(f, meshPoints):
	return np.asarray(map(lambda x: f(*x), meshPoints))

def bestfit(model, t, *params): return model(t, *params)



############## The Algorithm ###############

# 1. Create mesh of inputs
# (note: I don't like the whole spacing thing, don't think it works well
# or is even correct. Better to use number of points from the get-go)
def make_mesh(stepsize, dim):
	"""
	Input:	stepsize -- int or float, distance between mesh points
			dim -- int, dimension of space m

	Output:	meshPoints -- list, points of mesh arranged lexicographically
	"""

	# Define mesh
	numpts = (2. // stepsize) + 1.
	mesh = np.meshgrid(*[np.linspace( -1., 1., numpts) for i in xrange(dim)])

	# Make mesh into list of points
    # Order them lexicographically
	mesh = [mesh[i].reshape((np.prod(mesh[i].shape),)) for i in np.arange(len(mesh))]
	meshPoints = sorted(zip(*mesh))

	return meshPoints


# 2. Compute \nabla f(x) at mesh points and scale to unit length
# (note: cpIdx is an artifact of the old code -- I just made it smarter
# it's still dead though, no use for it anymore)
def compute_paths(gradf, meshPoints):
	"""
	Input:	gradf -- function, gradient of f: R^m -> R
			meshPoints -- list, input values in R^m

	Output:	gradPaths -- array, \nabla f evaluated at meshPoints, normalized
			cpIdx -- list, indices of critical points inside meshPoints
	"""

	# Evaluate grad f on points of base space
	gradPoints = np.asarray(map(lambda x: gradf(*x),meshPoints ))

	# Normalize
	gradPaths = preprocessing.normalize(gradPoints)

	# Find critical points, save them into an index list
	cpIdx = [i for i in xrange(len(gradPaths)) if np.linalg.norm(gradPaths[i]) == 0]

	return gradPaths, cpIdx


# 3'. Build the Active Manifold given function f
# (note: this is currently slow and dumb, could be made much better
# especially if nearest-neighbor search is done smarter)
def build_AM_given_f(seedPoint, meshPoints, f, gradPaths, stepsize):
	"""
	Input:	seedPoint -- array or list, random REGULAR value in [-1,1]^m (no CP)
			meshPoints -- list, points of mesh arranged vertically
			f -- function, f: R^m \to R
			gradPaths -- array, gradient evaluated at meshPoints
			stepsize -- float, specifies how large to step

	Output: activeManifold -- array, ordered points of the active manifold
			fValues -- array,
	"""

	# Define region / hypercube [-1,1]^(m+1)
	dim = len(meshPoints[0])
	rBound = np.ones(dim)
	Dom = spatial.Rectangle( -1*rBound, rBound )

	# Initialize activeManifold and fValues lists
	p0 = seedPoint

	# Find index of closest mesh point to p0
	# Use d0 for first direction
	i0 = spatial.KDTree(meshPoints).query(p0)[1]
	d0 = gradPaths[i0]

	# Initialize gradient ascent
	ascentPoints = np.asarray(p0)
	aRealVals = np.asarray(f(*p0))

	# Take one step
	p = p0 + (stepsize * d0)

	# Record results
	ascentPoints = np.vstack((ascentPoints,p))
	aRealVals = np.append(aRealVals, f(*p))

	cond = np.array(1)
	# Gradient Ascent
	while Dom.min_distance_point(ascentPoints[-1]) == 0 and min(cond.flatten()) > stepsize/3:

		i = spatial.KDTree(meshPoints).query(ascentPoints[-1])[1]
		d = gradPaths[i]

		p = ascentPoints[-1] + (stepsize * d)
		ascentPoints = np.vstack((ascentPoints, p))
		aRealVals = np.append(aRealVals, f(*p))

		#update loop condition
		cond = spatial.distance.cdist([ascentPoints[-1]], ascentPoints[0:len(ascentPoints)-1], 'euclidean')

	# Delete final elements (outside of hypercube)
	ascentPoints = np.delete(ascentPoints, len(ascentPoints) - 1, 0)
	aRealVals = np.delete(aRealVals, len(aRealVals) - 1, 0)

	# Initialize gradient descent
	descentPoints = np.asarray(p0)
	dRealVals = np.asarray(f(*p0))

	# Take one step
	p = p0 - (stepsize)*(d0)

	# Record Results
	descentPoints = np.vstack((descentPoints,p))
	dRealVals = np.append(dRealVals, f(*p))

	cond = np.array(1)
	# Gradient Descent
	while Dom.min_distance_point(descentPoints[-1]) == 0 and min(cond.flatten()) > stepsize/3:

		i = spatial.KDTree(meshPoints).query(descentPoints[-1])[1]
		d = gradPaths[i]

		p = descentPoints[-1] - (stepsize * d)
		descentPoints = np.vstack((descentPoints,p))
		dRealVals = np.append(dRealVals, f(*p))

		#update loop condition
		cond = spatial.distance.cdist([ascentPoints[-1]], ascentPoints[0:len(ascentPoints)-1], 'euclidean')

	# Delete first and last element in descentpoints and fValuesdescent
	descentPoints = np.delete(descentPoints, 0, 0)
	descentPoints = np.delete(descentPoints, len(descentPoints) - 1, 0)
	dRealVals = np.delete(dRealVals, 0)
	dRealVals = np.delete(dRealVals, len(dRealVals) - 1)

	# Flip order of descentpoints and concatenate lists
	descentPoints = np.flipud(descentPoints)
	dRealVals = np.flipud(dRealVals)

	activeManifold = np.concatenate((descentPoints, ascentPoints), axis=0)
	fValues = np.concatenate((dRealVals, aRealVals))

	return activeManifold, fValues


# 3. Build AM given samples of f-values and gradients
# (note: this is currently slow and dumb, could be made much better
# especially if nearest-neighbor search is done smarter)
def build_AM_from_data(seedPoint, meshPoints, fSamples, gradPaths, stepsize):
	"""
	Input:	seedPoint -- array or list, random REGULAR value in [-1,1]^m (no CP)
			meshPoints -- array, points of mesh arranged vertically
			f -- array, f evaluated at meshPoints
			gradDirections -- array, gradient evaluated at mesh points, normalized, arranged vertically
			stepsize -- float, specifies how large to step

	Output: activeManifold -- array, ordered points of the active manifold
			fValues -- array,
	"""

	# Define region / hypercube [-1,1]^(m+1)
	dim = len(meshPoints[0])
	rBound = np.ones(dim)
	#rBound = np.append(np.ones(liftDim - 1),t_0)
	Dom = spatial.Rectangle( -1*rBound, rBound )


	# Initialize activeManifold and fValues lists
	p0 = seedPoint

	# Find index of closest mesh point to seedpoint
	# Use d0 for first direction
	i0 = spatial.KDTree(meshPoints).query(p0)[1]
	d0 = gradPaths[i0]

	# Initialize gradient ascent
	ascentPoints = np.asarray(p0)
	aCloseVals = np.asarray(fSamples[i0])

	# Take one step
	p = p0 + (stepsize * d0)

	ascentPoints = np.vstack((ascentPoints,p))

	i1 = spatial.KDTree(meshPoints).query(p)[1]
	aCloseVals = np.append(aCloseVals, fSamples[i1])

	cond = np.array(1)
	# Gradient Ascent
	while Dom.min_distance_point(ascentPoints[-1]) == 0 and min(cond.flatten()) > stepsize/3:

		i = spatial.KDTree(meshPoints).query(ascentPoints[-1])[1]
		d = gradPaths[i]

		p = ascentPoints[-1] + (stepsize * d)
		ascentPoints = np.vstack((ascentPoints, p))

		i1 = spatial.KDTree(meshPoints).query(p)[1]
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

	i1 = spatial.KDTree(meshPoints).query(p)[1]
	dCloseVals = np.append(dCloseVals, fSamples[i1])

	cond = np.array(1)
	# Gradient Descent
	while Dom.min_distance_point(descentPoints[-1]) == 0 and min(cond.flatten()) > stepsize/3:

		i = spatial.KDTree(meshPoints).query(descentPoints[-1])[1]
		d = gradPaths[i]

		p = descentPoints[-1] - (stepsize * d)
		descentPoints = np.vstack((descentPoints,p))

		i1 = spatial.KDTree(meshPoints).query(p)[1]
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


# 4. Project points of interest to the AM
# (note: this is also slow and dumb. Mostly because I put a while loop condition
# in that's super expensive, just so I could get the damn thing to work.)
def project_to_AM(startPoint, meshPoints, activeManifold, gradPaths, stepsize):
	"""
	Input:	startPoint -- array or list, point in [-1,1]^m
			meshPoints -- list, points of mesh arranged vertically
			f -- function, user defined function
			activeManifold -- array, ordered points of the active manifold in R^m
			gradPaths -- array, grad f evaluated at meshPoints
			stepsize -- float, size of steps for the algorithm

	Output:	levSet -- array, points of level set with levSet[0] = startPoint, arranged vertically
			closeIndex -- int, index of closest point on active manifold to levSet[-1]
	"""

	# Initialize quantities of interest
	p0 = list(startPoint)
	closeIndex = 0
	levSet = list()

	levSet.append(p0)

	# Find index of closest point on active manifold to starting point
	manDist0, manInd0 = spatial.KDTree(activeManifold).query(p0)
	m0 = activeManifold[manInd0]

	# Check to see if starting point is already on the active manifold
	if manDist0 <= stepsize:
		closeIndex = manInd0
		print 'Point is nearly active manifold point %s' %(m0)
		levSet.append(list(m0))

	else:
		# Make vector m0 - p0
		v0 = m0 - p0

		# Find index of closest mesh point to p0 and its basePath
		meshDist0, meshInd0 = spatial.KDTree(meshPoints).query(p0)
		n0 = gradPaths[meshInd0]

		# Take a step along [x]
		w0 = v0 - np.dot(v0, n0)*n0

		d0 = w0 / np.linalg.norm(w0)
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

			#print "level set point %s" %(levSet[-1])
			closeIndex = manInd

			manDist, manInd = spatial.KDTree(activeManifold).query(levSet[-1])
			m = activeManifold[manInd]

			v = m - levSet[-1]

			meshDist, meshInd = spatial.KDTree(meshPoints).query(levSet[-1])
			n = gradPaths[meshInd]

			w = v - np.dot(v, n)*n
			d = w / np.linalg.norm(w)

			p = levSet[-1] + (stepsize * d)

			levSet.append(list(p))

			#update loop condition
			cond = spatial.distance.cdist([levSet[-1]], levSet[0:len(levSet)-1], 'euclidean')

	return np.vstack(levSet), closeIndex


#5. Find point of intersection between the level set and active manifold
def find_AM_intersection(levSet, activeManifold, closeIndex, meshPoints, gradPaths):
	"""
	Input:	levSet -- array, points of level set with levSet[0] = startPoint
			activeManifold -- array, ordered points of the active manifold
			closeIndex -- int, index of closest point on active manifold to levelSetPoints[-1]
			meshPoints -- array, points of mesh arranged vertically
			gradPaths -- array, gradient evaluated at mesh points, normalized, arranged vertically
			f -- function

	Output:	q -- float, approximate point of level set and active manifold intersection
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
	pIndex = spatial.KDTree(meshPoints).query(p)[1]
	n = gradPaths[pIndex]

	# Find closest point on active manifold to p
	m1 = activeManifold[closeIndex]

	# Delete m1 from active manifold to find second closest point
	cutAm = np.delete(activeManifold, closeIndex, 0)
	cutIndex = spatial.KDTree(cutAm).query(p)[1]
	m2 = cutAm[cutIndex]

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
	s = t * sValues[cutIndex] + (1 - t) * sValues[closeIndex]

	return q, s


# Function for plotting scipy best fit curve given a model function
def model_fit(model, activeManifold, fValues):
	"""
	Input:	model -- function to use for fit: f(x, *params)
			activeManifold -- array, points of active manifold, arranged vertically
			fValues -- array, function values of active manifold points

	Output: params -- array, optimal fit parameters for the model function
			covariance -- array, covariance matrix
	"""

	numpts = len(activeManifold)

	# Create sValues, evenly spaced in [0,1]
	sValues = np.linspace(0., numpts, numpts) / (numpts)

	# Fit fValues to a model function
	params, covariance = curve_fit(model, sValues, fValues)

	# Parameters for visualization
	xp = np.linspace(0., numpts, 2*(numpts)) / (numpts)
	plt.figure()

	plt.scatter(sValues,fValues, c='#66c2a5', s=30)
	plt.plot(xp, model(xp, *params), '-', c='#fc8d62', lw = 2)
	plt.xlim(0,1)
	plt.grid(True)
	plt.show(block=False)

	return params, covariance


# Load Data
# (note: idk what this does, but I think I used it maybe. It was here before me.)
def loadData(in_file):
	"""
	Load trial data via unjsonify and convert lists back to arrays

	Input:	in_file -- the file path where the object you want to read in is stored

	Output:	data -- dictionary of trial data, with lists converted back to arrays

	"""

	# Retrieve data
	data = unjsonify(in_file)

	# Convert lists back to arrays
	for key, value in data.iteritems():
		if type(value) == list:
			data[key] = np.array( data[key] )


	return data



######### Situational (really belongs in some kind of misc file)
# Note: This is just build_AM_given_f, mod 10000 in some spots and over the UHP
def build_AM_given_Loopy(seedPoint, meshPoints, f, gradPaths, stepsize):
	"""
	Construct Active Manifold from random starting point.
	Return points in active manifold and values of f


	Input:	seedpoint -- array or list, random point in [-1,1]^m (ex: np.ravel(2*np.random.rand(1, dim) - 1)
						**Don't Start Near a Critical Point**
			meshPoints -- array, points of mesh arranged vertically
			f -- function, user defined function
			gradDirections -- array, gradient evaluated at mesh points, normalized, arranged vertically
			stepsize -- float, same as gridspacing
			cpIdx  -- list, critical point index.

	Output: activeManifold -- array, ordered points of the active manifold
			fValues -- array,
	"""

	# Define region / hypercube [-1,1]^(m+1)
	dim = len(meshPoints[0])
	rBound = np.ones(dim)
	Dom = spatial.Rectangle( [-1,0], [1,1] )

	# Initialize activeManifold and fValues lists
	p0 = seedPoint

	# Find index of closest mesh point to p0
	# Use d0 for first direction
	i0 = spatial.KDTree(meshPoints).query(p0)[1]
	d0 = gradPaths[i0]

	# Initialize gradient ascent
	ascentPoints = np.asarray(p0)
	aRealVals = np.asarray(f(*p0))

	# Take one step
	p = p0 + (stepsize * d0)

	# Record results
	ascentPoints = np.vstack((ascentPoints,p))
	aRealVals = np.append(aRealVals, f(*p))

	cond = np.array(1)
	# Gradient Ascent
	while Dom.min_distance_point(ascentPoints[-1]) == 0 and min(cond.flatten()) > stepsize/3:

		i = spatial.KDTree(meshPoints).query(ascentPoints[-1])[1]
		d = gradPaths[i]

		p = ascentPoints[-1] + (stepsize * d)
		ascentPoints = np.vstack((ascentPoints, p))
		aRealVals = np.append(aRealVals, f(*p))

		#update loop condition
		cond = spatial.distance.cdist([ascentPoints[-1]], ascentPoints[0:len(ascentPoints)-1], 'euclidean')

	# Delete final elements (outside of hypercube)
	ascentPoints = np.delete(ascentPoints, len(ascentPoints) - 1, 0)
	aRealVals = np.delete(aRealVals, len(aRealVals) - 1, 0)

	# Initialize gradient descent
	descentPoints = np.asarray(p0)
	dRealVals = np.asarray(f(*p0))

	# Take one step
	p = p0 - (stepsize)*(d0)

	# Record Results
	descentPoints = np.vstack((descentPoints,p))
	dRealVals = np.append(dRealVals, f(*p))

	cond = np.array(1)
	# Gradient Descent
	while Dom.min_distance_point(descentPoints[-1]) == 0 and min(cond.flatten()) > stepsize/3:

		i = spatial.KDTree(meshPoints).query(descentPoints[-1])[1]
		d = gradPaths[i]

		p = descentPoints[-1] - (stepsize * d)
		descentPoints = np.vstack((descentPoints,p))
		dRealVals = np.append(dRealVals, f(*p))

		#update loop condition
		cond = spatial.distance.cdist([ascentPoints[-1]], ascentPoints[0:len(ascentPoints)-1], 'euclidean')

	# Delete first and last element in descentpoints and fValuesdescent
	descentPoints = np.delete(descentPoints, 0, 0)
	descentPoints = np.delete(descentPoints, len(descentPoints) - 1, 0)
	dRealVals = np.delete(dRealVals, 0)
	dRealVals = np.delete(dRealVals, len(dRealVals) - 1)

	# Flip order of descentpoints and concatenate lists
	descentPoints = np.flipud(descentPoints)
	dRealVals = np.flipud(dRealVals)

	activeManifold = np.concatenate((descentPoints, ascentPoints), axis=0)
	fValues = np.concatenate((dRealVals, aRealVals))

	return activeManifold, fValues
