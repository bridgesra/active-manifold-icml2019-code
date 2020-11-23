import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
from fractions import Fraction
from collections import OrderedDict

#dict for renaming columns for latex tables
col_name_dict = {'fitErrorL1': '$\ell^1$',
    'fitErrorL2': '$\ell^2$',
     'elapsed_time': 'time',
     'perc_testpt_used': '$n/N$',
     'm': '$m$',
     'n': '$n$'}

#change later so this is default names
f_name_dict = {'f1': '$f_1$',
    'f2': '$f_2$',
    'f3': '$f_3$',
    'uavg': '$u_{avg}$',
    'bind': '$B_{ind}$'}


#formatting funcs
def seconds_to_str(x):
    if x < .1:
        return str(1000*x)+'ms'
    elif x < 1.:
        return str(int(1000*x))+'ms'
    elif x < 100:
        return str(x)+'s'
    else:
        return str(int(x))+'s'

def convert_df_sigfigs(df, n_sig_figs = 3, min_sci_not = 1E-2):
    """
    converts all decimals in df to nicer format for display:
    decimals less than 1E-{sig_figs} are converted to sci not,
     otherwise converted to dec with 'sig_figs' signigican figures
    """
    #is_num_msk = df.applymap(np.isreal).all(0) #get numerical columns bool mask
    sci_not_fmt_str = "{0:.%dE}" %n_sig_figs
    dec_fmt_str = "{0:.%dg}" %n_sig_figs
    # df[is_num_msk] = df[is_num_msk].applymap(
    #     lambda x: sci_not_fmt_str.format(x).replace("E-0", "E-") if x < min_sci_not else dec_fmt_str.format(x))
    return df.applymap(
        lambda x: sci_not_fmt_str.format(x).replace("E-0", "E-") if x < min_sci_not else dec_fmt_str.format(x))
    return df

#io funcs
def write_table_to_csv_and_latex(df, outname, outdir, multicolumn_format = 'c|'):
    """
    Note requires booktab package (otherwise just replace *rule lines by \hline)
    """
    latex_filename = outname+'_latex.txt'; csv_filename = outname+'.csv'
    text_file = open(os.path.join(outdir,latex_filename), "w")
    text_file.write(df.to_latex(multirow=True, escape = False, multicolumn_format = multicolumn_format))
    text_file.close()
    df.reset_index().to_csv(os.path.join(outdir,csv_filename))
    print "wrote files %s, %s to %s" %(csv_filename, latex_filename, outdir)


#aggregating dataframe funcs
def get_benchmarking_summary_table(datafilepath, outdir, outname):

        df = pd.DataFrame.from_csv(datafilepath).reset_index()

        df.rename(columns = col_name_dict, inplace = True) #rename columns for latexing table

        #this is moronic and hacky and wont be necessary test_size is stored as ratio
        dec_to_ratio_dict = {'0.16667': 6, '0.33333': 3}
        df['Test Fraction'] = df.test_size.astype('str').apply(lambda x: Fraction(1,dec_to_ratio_dict[x]))

        #get summary stats and in good format for paper
        mean_df_grouped = df.groupby(['$m$', '$n$', 'Test Fraction', 'method']).agg(
            {'time':'mean'}).unstack().applymap(lambda x: float('%.3g' % x)) #get mean times and standardize sigfigs

        mean_df_grouped = mean_df_grouped.applymap(lambda x: seconds_to_str(x)) #fix s/ms formatting stuff
        mean_df_grouped.columns = mean_df_grouped.columns.map(' '.join)

        #output summary table to csv and latex
        write_table_to_csv_and_latex(df = mean_df_grouped, outname=outname, outdir=outdir)

def get_error_summary_table(datafilepath, outdir, outname,
    keep_cols = ['fitErrorL1', 'fitErrorL2','perc_testpt_used'],
     n_sig_figs = 3, min_sci_not = 1E-2):

    if not os.path.isfile(datafilepath):
        return None

    df = pd.DataFrame.from_csv(datafilepath).reset_index()
    df['perc_testpt_used'] = df.n_testpts_used/(df.n**df.m*df['test_size'])*100 #get num pts found by algo as perc of total test pts
    df.replace(f_name_dict, inplace = True) #replace function names for latex output
    df.rename(columns = col_name_dict, inplace = True)


    #get summary stats and in good format for paper
    ms_grouped = df.groupby(['f','method']).agg({
        '$\ell^1$':['mean', 'std'],
        '$\ell^2$':['mean', 'std'],
        'time':['mean', 'std'],
        '$n/N$':['mean']})

    #do some type conversion and formatting for latex output
    ms_grouped['time'] = ms_grouped['time'].applymap(lambda x: float('%.3g' % x)).applymap(lambda x: seconds_to_str(x))
    ms_grouped_sub = ms_grouped[[col_name_dict[col] for col in keep_cols]]
    ms_grouped_sub.columns = ms_grouped_sub.columns.map(' '.join)
    ms_grouped_out = convert_df_sigfigs(df = ms_grouped_sub,  n_sig_figs = n_sig_figs, min_sci_not = min_sci_not)
    is_num_msk =ms_grouped_out.applymap(np.isreal).all(0)
    write_table_to_csv_and_latex(df = ms_grouped_out, outname=outname, outdir=result_dir)

#make summary tables

if __name__ == '__main__':
    result_dir = os.path.abspath(os.path.join(os.getcwd(), 'ICML_results'))

    ####### Make Benchmarking Summary Table #########
    get_benchmarking_summary_table(datafilepath = os.path.join(result_dir, 'sum_squares_testruns.csv'),
        outname='sum_squares_timing_benchmark', outdir=result_dir)


    ####### Make Error Summary Table for 2d functions #########
    get_error_summary_table(datafilepath = os.path.join(result_dir, '2d_test_runs_AM_AS.csv'),
        outname='2d_test_function_error_summary', outdir=result_dir)

    # make error summary table for mhd example
    get_error_summary_table(datafilepath = os.path.join(result_dir, 'mhd_test_runs.csv'),
        outname='mhd_error_summary', outdir=result_dir, keep_cols = ['fitErrorL1', 'fitErrorL2'])
