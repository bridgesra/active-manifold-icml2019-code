import csv, os, sys
import numpy as np
import csv, os, sys
import time
import math
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'src')

import functions.ground.base_am_fxns as bf #anthonys base functions
import functions.fast_funcs as ff #mikis optimized base/main functions

#### functions
def run_test(method, dim, numPoints, f, grad_f, test_size, data_dict = None, stepsize = None,
    test_train_split_seed = 0, init_pt_seed = 0, seedPoint=None,
    outpath = None, return_all = False, verbose = False):
    """
    Prints and writes out info on each test run

    Input:  method -- str, method to test ('AM' or 'AS') (Active Manifold or Active Subspace)
            dim -- int, dimension of space
            numPoints -- number of points sampled unifomly along one axis of hypercube
            f -- function
            grad_f -- function (gradient function of f)
            test_size -- float in [0,1], fraction of test points in test/train split
            stepsize -- float, specifies how large to step (default= 2/3*(1/numPoints)*sqrt(dim))
            test_train_split_seed -- int, random seed for splitting into test/train set (default = 0)
            init_pt_seed -- int, seed for random initial point (default = 0) (ignored if SeedPoint is passed)
            seedPoint -- np.arr of len == dim, random REGULAR value in [-1,1]^dim (no CP)
                (pass none if using init_pt_seed for random start pt)
            outpath -- filepath, csv filepath to write results to (default = None)
            return_all -- if True (default = False) return dict of input params, results,
                and everything computed during algo run
                (for debugging or plotting, see if debug == True in fast_funcs.mainRandEx() for details on these vars)

    Output: rell long list of test run input params and results (written as row to csv if outpath is passed)
        [dim, numPoints, f (str name of function), stepsize, test_train_split_seed, method,
         seedPoint (as str), method,  l1 error (float), l2 error (float), runtime seconds (float), n_used_pts (int )*]
        *number of test points where algo successfully found path that intersected with manifold-- these are pts used in calculating error
    """
    #build data

    meshy, f_x, grads= ff.build_data(dim = dim, numPoints = numPoints, f = f, grad_f = grad_f)
    if method == 'AS' : grads = map(lambda x: grad_f(*x), meshy)

    return run_test_given_data(method = method, meshy=meshy, f_x= f_x, grads = grads,
        dim = dim, numPoints=numPoints, f_name = f.__name__, test_size=test_size, stepsize =stepsize,
        test_train_split_seed=test_train_split_seed, init_pt_seed =init_pt_seed, seedPoint=seedPoint,
        outpath=outpath, return_all = return_all, verbose = verbose )


def run_test_given_data(method, meshy, f_x, grads, dim, numPoints, f_name, test_size, stepsize = None,
    test_train_split_seed = 0, init_pt_seed = 0, seedPoint=None,
    outpath = None, return_all = False, verbose = False):
    """
    Prints and writes out info on each test run

    Input:  method -- str, method to test ('AM' or 'AS') (Active Manifold or Active Subspace)
            meshy -- array, points of mesh arranged vertically
            f_x-- array, function evaluated at meshPoints
            grads -- array, gradient evaluated at mesh points arranged vertically (for AM)
            dim -- int, dimension of space
            numPoints -- number of points sampled unifomly along one axis of hypercube
            f -- str, name of function
            test_size -- float in [0,1], fraction of test points in test/train split
            stepsize -- float, specifies how large to step (default= 2/3*(1/numPoints)*sqrt(dim))
            test_train_split_seed -- int, random seed for splitting into test/train set (default = 0)
            init_pt_seed -- int, seed for random initial point (default = 0) (ignored if SeedPoint is passed)
            seedPoint -- np.arr of len == dim, random REGULAR value in [-1,1]^dim (no CP)
                (pass none if using init_pt_seed for random start pt)
            outpath -- filepath, csv filepath to write results to (default = None)
            return_all -- if True (default = False) return dict of input params, results,
                and everything computed during algo run
                (for debugging or plotting, see if debug == True in fast_funcs.mainRandEx() for details on these vars)

    Output: rell long list of test run input params and results (written as row to csv if outpath is passed)
        [dim, numPoints, f (str name of function), stepsize, test_train_split_seed, method,
         seedPoint (as str), method,  l1 error (float), l2 error (float), runtime seconds (float), n_used_pts (int )*]
        *number of test points where algo successfully found path that intersected with manifold-- these are pts used in calculating error
    """

    #initialize default seedpoint and stepsize if not specified
    if seedPoint is None: seedPoint = ff.get_random_init_pt(seed = init_pt_seed, dim = dim) #get random start point
    if stepsize is None: stepsize = 2./3*(1./numPoints)*math.sqrt(dim) #make stepsize 2/3 length of longest diag in mesh

    rounded_seedpt = ff.multi_arg_map(round, seedPoint.tolist(), 4)
    #make list of test run input params for saving/printing
    param_list = [dim, numPoints, f_name, test_size, stepsize,
        test_train_split_seed, init_pt_seed, rounded_seedpt, method]

    if verbose:
        print ("method: %s, dim: %d, n: %d, f: %s, step size: %f, tt split seed: %d, seedPoint: %s"
            %(method, dim, numPoints, f_name, stepsize, test_train_split_seed, str(rounded_seedpt)))
        print "%d\%d train\\test pts" %(int(len(meshy)*(1-test_size)), int(len(meshy)*test_size))

    start_time = time.time()

    if method == 'AS':
        l1, l2 = ff.SubspEx(inputs=np.array(meshy),fSamples=f_x,grads=np.array(grads),
            test_size = test_size, test_train_split_seed =test_train_split_seed)
        n_testpts_used = int(len(meshy)*test_size) #all points used in constantines method

    elif method == 'AM':
        if return_all: #return everything calculated thru algo (for debugging and plotting)
                full_var_dict = ff.mainRandEx(
                        inputs=np.array(meshy),fSamples=f_x,grads=np.array(grads),
                        test_size = test_size, test_train_split_seed =test_train_split_seed,
                        stepsize=stepsize, seedPoint = seedPoint, debug = True)

                l1,l2, n_testpts_used = full_var_dict['fitErrorL1'], full_var_dict['fitErrorL2'], full_var_dict['n_testpts_used'] #results
                debug_vars = dict(list(full_var_dict.items())[:-3]) # all vars calculated during algo run

        else: #just return results
            l1, l2, n_testpts_used = ff.mainRandEx(
                inputs=np.array(meshy),fSamples=f_x,grads=np.array(grads),test_size = test_size,
                test_train_split_seed =test_train_split_seed, stepsize=stepsize, seedPoint = seedPoint)

    elapsed_sec = time.time()-start_time
    results = [l1, l2, n_testpts_used, elapsed_sec]

    if verbose:
        print 'L1 Error: %f, L2 Error: %f, Number Test Points Used: %d, Elapsed time: %fs' %tuple(results)

    if outpath: #write results to csv
        update_csv(data = param_list+results,
            header = ['m', 'n', 'f', 'test_size', 'stepsize', 'test_train_split_seed',
            'init_pt_seed', 'seedPoint', 'method', 'fitErrorL1', 'fitErrorL2',
            'n_testpts_used', 'elapsed_time'],
            filepath = outpath, multiple_rows= False)

    if return_all:
        return {'params': param_list, 'results': results, 'debug_var_dict' : debug_vars}
    return param_list+results




def update_csv(data, header, filepath, multiple_rows = True, overwrite = False):
    """
    Update/write new csv
    """
    file_exists = os.path.isfile(filepath)
    write_param = 'w' if overwrite else 'a'
    with open(filepath, write_param) as outcsv:
        writer = csv.writer(outcsv)
        if not file_exists or overwrite: writer.writerow(header)
        if multiple_rows:
            writer.writerows(data)
        else:
            writer.writerow(data)
    outcsv.close()


def plot_level_set_path_results(numPoints, f, grad_f, test_size, stepsize = None,
    test_train_split_seed = 0, init_pt_seed = 0, seedPoint=None, run_dict = None, error_threshold = 10, outpath = None):
    """
    plots pretty pictures

    dim -- int, dimension of space
    numPoints -- number of points sampled unifomly along one axis of hypercube
    f -- function
    grad_f -- function (gradient function of f)
    test_size -- float in [0,1], perc of test points in test/train split
    stepsize -- float, specifies how large to step (default= 2/3*(1/numPoints)*sqrt(dim))
    test_train_split_seed -- int, random seed for splitting into test/train set (default = 0)
    init_pt_seed -- int, seed for random initial point (default = 0) (ignored if SeedPoint is passed)
    d -- dict, dict containing all results from algo run (all above params are ignored)
    seedPoint -- np.arr of len == dim, random REGULAR value in [-1,1]^dim (no CP)
        (pass none if using init_pt_seed for random start pt)
    outpath -- filepath (.png, .pdf, ect.), path to write figure to (default = None)

    Output: matplotlib figure

    """
    d = run_test(method = 'AM', dim = 2, numPoints=numPoints, f =f, grad_f=grad_f,
            stepsize = stepsize,test_train_split_seed=test_train_split_seed,
             init_pt_seed = init_pt_seed,  test_size=test_size, seedPoint =seedPoint,
            verbose = True, return_all = True)

    fig = draw_plots( d = d['debug_var_dict'], error_threshold = error_threshold, outpath =outpath)

def draw_plots(d,error_threshold = 10, outpath = None ):
    activeManifold, ptwise_error =  d['activeManifold'], d['ptwise_error']
    paths, non_intersecting_paths, no_intersection_msk = d['paths'], d['non_intersecting_paths'], d['no_intersection_msk']

    bad_pt_paths = [np.dstack(paths[idx])[0] for idx in  np.where(ptwise_error>error_threshold)[0]] #get paths of high error pts
    good_pt_paths = [np.dstack(paths[idx])[0] for idx in  np.where(ptwise_error<=error_threshold)[0]]#get paths of low error pts
    non_intersect_pts = zip(*[p[0] for p in non_intersecting_paths])

    fig, ax = plt.subplots(figsize=(10,10))
    for path in good_pt_paths: #plot levelsets for good pts
        ax.plot(path[0][0],path[1][0],'bx')
        ax.plot(path[0],path[1],'b.-', markersize = .1)
    for path in bad_pt_paths: #plot levelsets for bad pts
        ax.plot(path[0][0],path[1][0],'rx')
        ax.plot(path[0],path[1],'r.-', markersize = .5)
    ax.plot(*zip(*activeManifold), color = 'k', lw = 2)#plot manifold
    if non_intersect_pts: ax.plot(non_intersect_pts[0], non_intersect_pts[1], 'x')

    if outpath:
        fig.savefig(outpath)
    return fig
