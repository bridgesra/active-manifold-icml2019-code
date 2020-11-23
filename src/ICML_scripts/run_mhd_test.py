import warnings
warnings.filterwarnings('ignore')

import numpy as np
import csv, os, sys
import time
import argparse

sys.path.insert(0, 'src')

import functions.ground.base_am_fxns as bf
from functions.testing_plotting_functions import *
from functions.mhd_functions import *

# To recreate error tables for MHD example in in paper submission
#   1. Run this script 2 times as shown in A., B. below
#   2. Run make_tables script:
#       `python src/ICML_scripts/make_tables.py`
#

####
# Script for running tests MHD example for paper submission
# This script creates a mesh of 100,000 pts uniformly sampled in  [-1,1]^5 (2000 on each axis)
# and function values according to function name passed by user (uavg or bind, see src/functions/mhd_functions.py for details)
# it then picks 3 random start points and runs AM and AS code building manifold on 3 different train/test splits
# where ratio of train:test data is based on user passed test ratio
# and then runs both AS/AM code on each of these splits
# it then prints test run params and results (l1/l2 error, runtime, n test pts found by AM algo)
# and writes results to specifed filename
# For more detailed info on user passed params run ` python src/ICML_scripts/run_mhd_test.py --h` or see below

##### To reproduce MHD error table in paper submission ########

# In order to minimize runtime run each of these on a different core (run in new terminal window)
# Pass --w to make sure each does not try and write to same file simultaneously
## A. Run AS and AM algo with function u_avg (average flow velocity) with 1/50 test data
# python src/ICML_scripts/run_mhd_test.py --f ICML_results/mhd_test_runs.csv --r 50 --l uavg --w

## B. Run AS and AM algo with function B_ind (induced magnetic field) with 1/50 test data
# python src/ICML_scripts/run_mhd_test.py --f ICML_results/mhd_test_runs.csv --r 50 --l bind --w

####




#####parse command line args
argparser = argparse.ArgumentParser(description='run tests on AM and AS algos on mhd example')
argparser.add_argument('--filepath', type = str,
    default = None,
    help = 'filename (csv) to output results to')
argparser.add_argument('--ratio_test', type = int,
    help = 'pass ratio of test to train points')
argparser.add_argument('--list_functions', nargs='+', help='list of test function names (uavg, bind)')
# argparser.add_argument('--stepsize', type = float,default = None,
#     help = "stepsize (default= .15)"
argparser.add_argument('--wait', action = 'store_true',
    help = 'pass flag to wait to write to file until end (pass if running multiple instances of this script writing to same file)')
argparser.add_argument('--overwrite', action = 'store_true',
    help = 'pass to overwrite dat in filename if already exists (only pass with --wait flag)')

try:
    args = argparser.parse_args()
except:
    sys.exit(1)

function_list = args.list_functions
#this would have been better to save as integer num and denom
#and pass test_size num./denom to main functions (change in nextversion)
test_size = round(1./args.ratio_test,5)
outfilepath = args.filepath #get passed out filepath
wait = args.wait #bool indicating whether to wait to write to file until end
overwrite = args.overwrite

####test
if __name__ == '__main__':

    #save results to this folder
    outdir = os.path.dirname(outfilepath)
    test_run_results_dir = os.path.abspath(os.path.join(os.getcwd(), outdir))
    if not os.path.isdir(test_run_results_dir):
        os.makedirs(test_run_results_dir)

    outpath = os.path.join(test_run_results_dir, outfilepath.split('/')[-1]) #get absolute path of outfile
    single_run_outpath = outpath if not wait else None #if wait to write set to none

    # #run test over specified function, 3 random initial points 3 randpme test/train splits
    # and write out run info, l1 error, l2 error, runtime, and number of test points successfully found by algo (for AM only)
    # for AM and AS methods

    meshy = ff.make_mesh(dim = 5, step = .2)

    result_list = []
    for func_name in function_list:
        if func_name == 'uavg':
            f_x, grads, df = get_u_avg_data(meshy)
        elif func_name == 'bind':
            f_x, grads, df = get_b_ind_data(meshy)

        for init_seed in list(range(0,3)):
            for tt_seed in list(range(0,3)):
                for method in ['AS','AM']:
                    grads = df if method == 'AS' else grads
                    results = run_test_given_data(method = method, meshy = meshy,f_x = f_x, grads= grads,
                     	dim = 5, numPoints = 100, stepsize = .15, f_name = func_name, test_size = test_size,
                    	test_train_split_seed=tt_seed, init_pt_seed = init_seed, outpath= single_run_outpath, verbose = True)
                    result_list.append(results)
                    print('\n')

    if wait: #if waited to write until ennd
        update_csv(data = result_list,
            header = ['m', 'n', 'f', 'test_size', 'stepsize',
            'test_train_split_seed','init_pt_seed', 'seedPoint', 'method',
            'fitErrorL1', 'fitErrorL2', 'n_testpts_used', 'elapsed_time'],
            filepath =outpath, overwrite=overwrite)
