import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os, sys
import math
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy import spatial
from sklearn import model_selection

sys.path.insert(0, 'src')
sys.path.insert(0, 'active_subspaces')

import functions.ground.base_am_fxns as bf #anthonys base functions
import functions.fast_funcs as ff #mikis optimized base/main functions
import functions.plotfuncs as plotfuncs
import active_subspaces as ac


if __name__ == '__main__':

    outdir = os.path.abspath(os.path.join(os.getcwd(), 'ICML_results'))
    if not os.path.isdir(outdir):
        os.makedirs(outdir)


    dim = 2; f = bf.f3; g = bf.gradf3
    init_pt_seed = 0

    meshy = ff.make_mesh(dim,.01)
    f_x = bf.sample_f_on_mesh(f,meshy)
    amGrads = bf.compute_paths(g,meshy)[0]

    seedPoint = ff.get_random_init_pt(seed = init_pt_seed, dim = dim)

    stepsize = 2./3*(1./int(math.sqrt(len(meshy))))*math.sqrt(dim)

    # Compute gradients at trainingPoints
    gradPaths = preprocessing.normalize(amGrads)

    # Build KD Tree for mesh
    mesh_kdtree = spatial.KDTree(meshy)

    # Build active manifold from training points
    activeManifold, fCloseVals = ff.build_AM_from_data(
        seedPoint = seedPoint, mesh_kdtree = mesh_kdtree,
        fSamples= f_x, gradPaths=gradPaths, stepsize=stepsize)

    fig = plotfuncs.splinefitf3(activeManifold, fCloseVals)
    fig.savefig(os.path.join(outdir,'f3-AM.pdf'),bbox_inches = 'tight')

    grads = np.array(map(lambda x: g(*x), meshy))


    #Taken from SubspEx() in fast_funcs.py
    X_train, X_test, y_train, y_test, grad_train, grad_test= model_selection.train_test_split(
        meshy, f_x, grads, test_size = 8000, random_state = 0)

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
    plt.figure(figsize=(7, 7))
    y0 = np.linspace(-1, 1, 200)
    plt.scatter(y, y_train, c = '#66c2a5')
    plt.plot(y0, RS.predict(y0[:,None])[0], c = '#fc8d62', ls='-',linewidth=2)
    plt.grid(True)
    plt.xlabel('Active Variable Value', fontsize=18)
    plt.ylabel('Output', fontsize=18)
    plt.savefig(os.path.join(outdir,'f3-AS.pdf'),bbox_inches = 'tight')
