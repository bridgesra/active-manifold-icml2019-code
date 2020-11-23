import warnings
warnings.filterwarnings('ignore')

import numpy as np
import csv, os, sys
import time
import argparse

sys.path.insert(0, 'src')

import functions.ground.base_am_fxns as bf
from functions.testing_plotting_functions import *

# To recreate error/benchmarking tables test/benchmarking functions in in paper submission
#   1. Run this script 8 times as shown in A., B. below
#   2. Run make_tables script:
#       `python src/ICML_scripts/make_tables.py`
#

####
# Script for running 9 test runs for each test/benchmarking functions in paper submission
# This script creates a mesh of data and function values according to params passed by User (m,n, list of fs)
# it then picks 3 random start points and runs AM and AS code building manifold on 3 different train/test splits
# where ratio of train:test data is based on user passed test ratio
# and then runs both AS/AM code on each of these splits
# it then prints test run params and results (l1/l2 error, runtime, n test pts found by AM algo)
# and writes results to specifed filename
# For more detailed info on user passed params run ` python src/ICML_scripts/run_tests.py --h` or see below

##### To reproduce error and benchmarking test results in paper submission ########

# FOR ERROR TABLE:
# Run on 100x100 pts in [-1,1]^2 cube with 20% test data over test functions f1, f2, f3 as defined in paper:
#   python src/ICML_scripts/run_tests.py --n 100 --m 2 --r 5 --l f1 f2 f3 --f ICML_results/2d_test_runs_AM_AS.csv

# FOR BENCHMARKING TABLE:
# In order to minimize runtime run each of these on a different core (run in new terminal window)
# 2d runs fast enoguht to make this uncessary, but this is useful for 3d funcs
# Pass --w to make sure each does not try and write to same file simultaneously
## A. Run 4 test runs on 2 dims: run on 15x15 and 30x30 pts in [-1,1]^2 square with 1/6 and 1/3 test data over sum of squares function f(x,y) = |x|^2 + |y|^2:
#    python src/ICML_scripts/run_tests.py --n 15 --m 2 --l ss --r 6 --f ICML_results/sum_squares_testruns.csv --w
#    python src/ICML_scripts/run_tests.py --n 15 --m 2 --l ss --r 3 --f ICML_results/sum_squares_testruns.csv --w
#    python src/ICML_scripts/run_tests.py --n 30 --m 2 --l ss --r 6 --f ICML_results/sum_squares_testruns.csv --w
#    python src/ICML_scripts/run_tests.py --n 30 --m 2 --l ss --r 3 --f ICML_results/sum_squares_testruns.csv --w

## B. Run 4 test runs on 3 dims: run on 15x15x15 and 30x30x30 pts in [-1,1]^3 cube with 1/6 and 1/3 test data over sum of squares function f(x,y, z) = |x|^2 + |y|^2 + |z|^2:
#    python src/ICML_scripts/run_tests.py --n 15 --m 3 --l ss --r 6 --f ICML_results/sum_squares_testruns.csv --w
#    python src/ICML_scripts/run_tests.py --n 15 --m 3 --l ss --r 3 --f ICML_results/sum_squares_testruns.csv --w
#    python src/ICML_scripts/run_tests.py --n 30 --m 3 --l ss --r 6 --f ICML_results/sum_squares_testruns.csv --w
#    python src/ICML_scripts/run_tests.py --n 30 --m 3 --l ss --r 3 --f ICML_results/sum_squares_testruns.csv --w

####

#map for easy calling of test function in `functions.ground.base_am_fxns`:
#function name: (function, grad) tuple
test_function_dict = {
    'f1': (bf.f1, bf.gradf1),
    'f2': (bf.f2, bf.gradf2),
    'f3': (bf.f3, bf.gradf3),
    'ss': (bf.Squaresum, bf.gradSquaresum)}

#####parse command line args
argparser = argparse.ArgumentParser(description='run tests on AM and AS algos on test data')
argparser.add_argument('--filepath', type = str,
    default = None,
    help = 'filename (csv) to output results to')
argparser.add_argument('--numpts', type = int,
    help = 'pass number of points along 1 axis of hypercube')
argparser.add_argument('--m', type = int,
    help = 'pass dimension of hypercube')
argparser.add_argument('--ratio_test', type = int,
    help = 'pass ratio of test to train points')
argparser.add_argument('--list_functions', nargs='+', help='list of test function names (f1, f2, f3, ss)')
argparser.add_argument('--wait', action = 'store_true',
    help = 'pass flag to wait to write to file until end (pass if running multiple instances of this script writing to same file)')
argparser.add_argument('--overwrite', action = 'store_true',
    help = 'pass to overwrite dat in filename if already exists (only pass with --wait flag)')

try:
    args = argparser.parse_args()
except:
    sys.exit(1)

numPoints = args.numpts
dim = args.m
function_list = args.list_functions
#this would have been better to save as integer num and denom
#and pass test_size num./denom to main functions (change in nextversion)
test_size = round(1./args.ratio_test,5)
outfilepath = args.filepath #get passed out filepath
wait = args.wait #bool indicating whether to wait to write to file until end
overwrite = args.overwrite


if __name__ == '__main__':

    #save results to this folder
    outdir = os.path.dirname(outfilepath)
    test_run_results_dir = os.path.abspath(os.path.join(os.getcwd(), outdir))
    if not os.path.isdir(test_run_results_dir):
        os.makedirs(test_run_results_dir)

    outpath = os.path.join(test_run_results_dir, outfilepath.split('/')[-1]) #get absolute path of outfile
    single_run_outpath = outpath if not wait else None #if wait to write set to none

    # #run test over different functions, 3 random initial points 3 randpme test/train splits
    # and write out run info, l1 error, l2 error, runtime, and number of test points successfully found by algo (for AM only)
    # for AM and AS methods
    result_list = []
    for func_name in function_list:
        f, g = test_function_dict[func_name]
        for init_seed in list(range(0,3)):
            for tt_seed in list(range(0,3)):
                for method in ['AS','AM']:
                    results = run_test(method = method, dim = dim, numPoints=numPoints, f = f, grad_f=g,
                        stepsize = None,test_train_split_seed=tt_seed, init_pt_seed = init_seed, test_size= test_size,seedPoint = None,
                        verbose = True, outpath= single_run_outpath)
                    result_list.append(results)

    if wait: #if waited to write until ennd
        update_csv(data = result_list,
            header = ['m', 'n', 'f', 'test_size', 'stepsize',
            'test_train_split_seed','init_pt_seed', 'seedPoint', 'method',
            'fitErrorL1', 'fitErrorL2', 'n_testpts_used', 'elapsed_time'],
            filepath =outpath, overwrite=overwrite)
