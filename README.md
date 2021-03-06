# README for Active Manifolds ICML 2019 Code

## Copyright info: 
This code has an open source [copyright](./Open\ Source\ Software\ License\ (3-Clause).doc)


## Please cite the [ICML Publication](http://proceedings.mlr.press/v97/bridges19a.htmlhttp://proceedings.mlr.press/v97/bridges19a.html) for use of this code. 

	@InProceedings{bridges2019active, 
		title = {Active Manifolds: A non-linear analogue to Active Subspaces}, 
		author = {Bridges, Robert and Gruber, Anthony and Felder, Christopher and Verma, Miki and Hoff, Chelsey}, 
		pages = {764--772}, 
		year = {2019}, 
		editor = {Kamalika Chaudhuri and Ruslan Salakhutdinov}, 
		volume = {97}, 
		series = {Proceedings of Machine Learning Research}, 
		address = {Long Beach, California, USA}, month = {09--15 Jun}, 
		publisher = {PMLR}, 
		pdf = {http://proceedings.mlr.press/v97/bridges19a/bridges19a.pdf}, 
		url = {http://proceedings.mlr.press/v97/bridges19a.html}, 
		abstract = {We present an approach to analyze $C^1(\mathbb{R}^m)$ functions that addresses limitations present in the Active Subspaces (AS) method of Constantine et al. (2014; 2015). Under appropriate hypotheses, our Active Manifolds (AM) method identifies a 1-D curve in the domain (the active manifold) on which nearly all values of the unknown function are attained, which can be exploited for approximation or analysis, especially when $m$ is large (high-dimensional input space). We provide theorems justifying our AM technique and an algorithm permitting functional approximation and sensitivity analysis. Using accessible, low-dimensional functions as initial examples, we show AM reduces approximation error by an order of magnitude compared to AS, at the expense of more computation. Following this, we revisit the sensitivity analysis by Glaws et al. (2017), who apply AS to analyze a magnetohydrodynamic power generator model, and compare the performance of AM on the same data. Our analysis provides detailed information not captured by AS, exhibiting the influence of each parameter individually along an active manifold. Overall, AM represents a novel technique for analyzing functional models with benefits including: reducing $m$-dimensional analysis to a 1-D analogue, permitting more accurate regression than AS (at more computational expense), enabling more informative sensitivity analysis, and granting accessible visualizations (2-D plots) of parameter sensitivity along the AM.} }
 


## Notes:
- Python 2.7 needed along w/ appropriate packages. If an `import <packagename>`  fails then installation of that package is needed.
- All command line example assumes running from inside `active-manifolds/`, and writes to folder `ICML_results/`
- All plotting code below is independent of the tables / code above


### Make benchmarking and error result tables (Table 1, 3, 4):
1. Run tests in `src/ICML_scripts/run_tests.py` as described in header comment. See code below:
	*  FOR BENCHMARKING TABLE (Table in Supplementary Section):
		* Writes results to csv filepath passed to `--f` argument: example shown below writes to `sum_squares_testruns.csv`
		* In order to minimize runtime run each of these on a different core (run in new terminal window)
		* While 2d function runs in (A) run fast enough to make this uncessary, but this is useful for 3d funcs in (B).
		*  If running concurrently, pass `--w` flag to make sure each does not try and write to same file simultaneously.
		*  Warning: Last 2 runs of (B) take upwards of 15 min each


		**A.** Run the 4 test runs on 2 dims: run on 15x15 and 30x30 pts in $[-1,1]^2$ square with 1/6 and 1/3 test data over sum of squares function $f(x,y) = |x|^2 + |y|^2$:


			python src/ICML_scripts/run_tests.py --n 15 --m 2 --l ss --r 6 --f ICML_results/sum_squares_testruns.csv --w
			python src/ICML_scripts/run_tests.py --n 15 --m 2 --l ss --r 3 --f ICML_results/sum_squares_testruns.csv --w
			python src/ICML_scripts/run_tests.py --n 30 --m 2 --l ss --r 6 --f ICML_results/sum_squares_testruns.csv --w
			python src/ICML_scripts/run_tests.py --n 30 --m 2 --l ss --r 3 --f ICML_results/sum_squares_testruns.csv --w


		**B.** Run 4 test runs on 3 dims: run on 15x15x15 and 30x30x30 pts in $[-1,1]^3$ cube with 1/6 and 1/3 test data over sum of squares function $f(x,y, z) = |x|^2 + |y|^2 + |z|^2$:
		
			python src/ICML_scripts/run_tests.py --n 15 --m 3 --l ss --r 6 --f ICML_results/sum_squares_testruns.csv --w
			python src/ICML_scripts/run_tests.py --n 15 --m 3 --l ss --r 3 --f ICML_results/sum_squares_testruns.csv --w
			python src/ICML_scripts/run_tests.py --n 30 --m 3 --l ss --r 6 --f ICML_results/sum_squares_testruns.csv --w
			python src/ICML_scripts/run_tests.py --n 30 --m 3 --l ss --r 3 --f ICML_results/sum_squares_testruns.csv --w


	* FOR 2D TEST FUNCTION ERROR TABLES (Table 1):
		* Writes results to csv filepath passed to `--f` argument: example shown below writes to `2d_test_runs_AM_AS.csv`
		* This also runs slowly, so to run more quickly, run each function seperately as shown below. To run serially, pass `--l f1 f2 f3` instead, in which case the `--w` wait flag is no longer necessary

		**C.** Run on 100x100 pts in [-1,1]^2 cube with 20% test data over test functions f1, f2, f3 as defined in paper:
		
			python src/ICML_scripts/run_tests.py --n 100 --m 2 --r 5 --l f1 --f ICML_results/2d_test_runs_AM_AS.csv --w
			python src/ICML_scripts/run_tests.py --n 100 --m 2 --r 5 --l f2 --f ICML_results/2d_test_runs_AM_AS.csv --w
			python src/ICML_scripts/run_tests.py --n 100 --m 2 --r 5 --l f3 --f ICML_results/2d_test_runs_AM_AS.csv --w
		

2. Run tests in `run_mhd_test.py` as described in header comment

	* FOR MHD EXAMPLE ERROR TABLE (Table 3):
		* Writes results to csv filepath passed to `--f` argument: example shown below writes to `mhd_test_runs.csv`
		* See `src/functions/mhd_functions.py `for details on functions called

		**A.** Run on 100,0000 pts uniformly sampled over in [-1,1]^5 cube with 2% test data over u_avg (average flow velocity) function as described in paper:
	

			python src/ICML_scripts/run_mhd_test.py --f ICML_results/mhd_test_runs.csv --r 50 --l uavg --w
		

		**B.** Run on 100,0000 pts uniformly sampled over in [-1,1]^5 cube with 2% test data over B_ind (induced magnetic field) function as described in paper:
		
			python src/ICML_scripts/run_mhd_test.py --f ICML_results/mhd_test_runs.csv --r 50 --l bind --w

2. Run `make_tables.py` to create summary tables for these test runs:
	* Writes summary table (in csv and latex format) for benchmarking table made in 1A-B to `sum_squares_timing_benchmark_latex.txt*, `sum_squares_timing_benchmark.csv`
	* Writes summary table (in csv and latex format) for error table made in 1C to `2d_test_function_error_summary_latex.txt`, `2d_test_function_error_summary.csv`
	*  Writes summary table (in csv and latex format) for error table made in 2A-B to `mhd_error_summary_latex.txt`, `mhd_error_summary.csv`


		python src/ICML_scripts/make_tables.py



### Make level set plots (Fig. 2)
* Run `make_2d_func_AM_traversal_plots.py` to make 3 level set plots for each of the test functions f1, f2, f3
* Writes figures to `f1_eg_plot_500pts.png`, `f2_eg_plot_500pts.png`, `f3_eg_plot_500pts.png`


		python src/ICML_scripts/make_2d_func_AM_traversal_plots.py
	

#### Make $f_3$ AM and AS plots (Fig. 4)
* Run `make_f3_plot.py` to create plots of f3 along AM and AS.
* Writes figures to: `f3-AM.pdf`, `f3-AS.pdf`

		python src/ICML_scripts/make_f3_plot.py

#### Make MHD plots (Fig. 5,6)

* Run `make_MHD_plots.py` to recreate 8 plots for MHD problem
* Writes figures to: `Hartmann_BDerivs.pdf`, `Hartmann_uFitSpline.pdf`, `MHD_uDerivs.pdf`, `Hartmann_BFitSpline.pdf`, `MHD_BDerivs.pdf`       `MHD_uFitSpline.pdf`, `Hartmann_uDerivs.pdf`, `MHD_BFitSpline.pdf`


		python src/ICML_scripts/make_MHD_plots.py
