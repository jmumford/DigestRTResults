# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3.8.5 64-bit ('3.8.5')
#     metadata:
#       interpreter:
#         hash: 894fa685f23f60d5a1280e0272d4ff76a9e0a58e4d9f83d19fa9a7ac0ff492c7
#     name: python3
# ---

# +
# practice with just one design.mat
import numpy as np
from nilearn.plotting import plot_stat_map
from nilearn.image import math_img
import json
from pathlib import Path
import pandas as pd


# Function to threshold stat image
def mask_tstat_img(tstat_path, p_gt0_path, p_lt0_path):
    """Function for thresholding t-statistic map based on two 1-sided
    1-p values maps.  Output is thresholded t-stat map
    tstat_path:  Path to t-stat image
    p_gt0_path:  Path to 1-p value image for contrast>0
    p_lt0_path:  Path to 1-p value image for contrast<0
    """
    thresh_tstat_img = math_img('((img1 > 0.95) + (img2 > 0.95)) * img3',
                                img1=str(p_gt0_path), img2=str(p_lt0_path),
                                img3=str(tstat_path))
    return thresh_tstat_img


def get_json_contents(json_file):
    """
    load contents from the task contrast json
    """
    with open(json_file) as f:
        return(json.load(f))


def get_contrast_keys_from_json_contents(contrast_names, search_key='RT'):
    return([key for key, val in contrast_names.items() if search_key in val])


def search_analysis_make_figures(taskdir):
    """Function that loops through all analysis directories (randomise) within a
    task and finds contrasts involving RT as a covariate.  If no significant
    results, message is output to screen.  Otherwise a thresholded t-stat map
    (p<0.05) is created.
    taskname: name of task.  One of ANT, CCTHot, DPX, WATT3, discountFix,
                            motorSelective Stop,stopSignal, stroop and twoByTwo
    """

    taskname = taskdir.parts[-2]
    analysis_dirs = taskdir.glob('*Randomise')
    for current_directory in analysis_dirs:
        json_file = current_directory / 't_name_map.json'
        contrast_dict = get_json_contents(json_file)
        contrast_keys = get_contrast_keys_from_json_contents(contrast_dict)

        if len(contrast_keys) % 2 == 0:
            num_tstats = len(contrast_keys) // 2
        else:
            print('skipping {0} since odd number of RT contrasts'.
                  format(current_directory))
            continue
     
        for connum in range(num_tstats):
            pos_key = contrast_keys[2 * connum]
            neg_key = contrast_keys[2 * connum + 1]

            dependent_variable_name = current_directory.name
            # remove last three letters which specify Pos/Neg
            independent_variable_name = contrast_dict[pos_key][:-3]

            # Threshold t-stat image
            tstat_path = current_directory / \
                f"randomise_tstat{pos_key}.nii.gz"
            p_gt0_path = current_directory / \
                f"randomise_tfce_corrp_tstat{pos_key}.nii.gz"
            p_lt0_path = current_directory / \
                f"randomise_tfce_corrp_tstat{neg_key}.nii.gz"
            thresh_tstat = mask_tstat_img(tstat_path, p_gt0_path, p_lt0_path)

            # Only create image if there are significant voxels
            sig_voxels = np.count_nonzero(thresh_tstat.dataobj)

            if sig_voxels > 0:
                print(f"Yes significant results (map below): cor({dependent_variable_name}, {independent_variable_name})")
                plot_stat_map(thresh_tstat,
                              title=f"{taskname}:{dependent_variable_name} correlated with {independent_variable_name}",
                              display_mode='z', threshold=0, cut_coords=10,
                              black_bg=True)
            else:
                print(f"No significant results: cor({dependent_variable_name}, {independent_variable_name})")


def convert_json_to_design_matrix(randomise_dir_json_file):
    """ Within each randomise directory is a json file 
        starting with "_", followed by a series of characters
        I'm assuming nipype created it, but I'm not sure
        This function reads in the json file and returns
        the design matrix as a pandas data frame
        Input: Path to the json file that starts with "_"
        Output: Pandas data frame containing all regressors
    """
    with open(randomise_dir_json_file) as f:
        randomise_info = json.load(f)
    randomise_information_dict = {randomise_info_subarray[0]: 
        randomise_info_subarray[1] for randomise_info_subarray in randomise_info}
    design_matrix_dict = {design_info[0]: design_info[1] for 
            design_info in randomise_information_dict['regressors']}
    regressor_names = design_matrix_dict.keys()
    design_matrix = pd.DataFrame(design_matrix_dict)
    return(design_matrix.astype(float))


        
def get_regressor_correlations(taskdir):
    """ Used to check if the regressor names and values match
        for all deign matrices across randomise analyses
        selected within the specific task
        
        Output: Print regressor names and correlation of regressors 
            only prints multiples if differences occur between
            randomise directories within taskdir
    """
    analysis_dirs = [i for i in taskdir.glob('*Randomise')]

    for index, current_directory in enumerate(analysis_dirs):
        design_json_path = [i for i in current_directory.glob("_*.json")]
        dependent_variable_name = design_json_path[0].parts[-2]
        if index == 0:
            design_matrix_first = convert_json_to_design_matrix(design_json_path[0])
            design_first_dependent_variable_name = dependent_variable_name
            print(f"Correlation of regressors for: {design_first_dependent_variable_name}")
            print(design_matrix_first.corr().round(decimals=3))
        if index > 0:
            design_matrix_not_first = convert_json_to_design_matrix(design_json_path[0])
            if design_matrix_first.equals(design_matrix_not_first):
                print(f"Design for {dependent_variable_name} matches {design_first_dependent_variable_name}")
            else:
                print(f"Correlation of regressors for: {dependent_variable_name}")
                print(design_matrix_first.corr().round(decimals=3))


def count_analysis_dirs_and_reaction_time_contrasts(taskdir):
    """ Count number of analysis directories within a task and RT-related contrasts
        Output:  Prints message indicating both of these numbers
        If number of RT related contrasts differs for some designs,
        this will be indicated
    """
    analysis_dirs = taskdir.glob('*Randomise')
    contrast_count = {}
    for current_directory in analysis_dirs:
        json_file = current_directory / 't_name_map.json'
        contrast_dict = get_json_contents(json_file)
        contrast_keys = get_contrast_keys_from_json_contents(contrast_dict)

        if len(contrast_keys) % 2 == 0:
            num_tstats = len(contrast_keys) // 2
        else:
            print('skipping {0} since odd number of RT contrasts'.
                  format(current_directory))
            continue
        contrast_name = current_directory.parts[-1]
        contrast_count[contrast_name] = num_tstats
    num_analysis_directories = len(contrast_count)
    num_rt_contrasts = list(contrast_count.values())
    if num_rt_contrasts.count(num_rt_contrasts[0]) == len(num_rt_contrasts):
        print(f"***There are {num_analysis_directories} randomise analyses each with {num_rt_contrasts[0]} RT-related contrasts (pos and neg)***")
    else:
        print(f"***There are {num_analysis_directories} randomise analyses with variable numbers of RT-related contrasts as indicated below***") 
        print(contrast_count)



# + tags=[]
basedir = Path('/Users/jeanettemumford/sherlock_local/uh2/aim1/BIDS_scans/\
derivatives/2ndlevel_4_2_21')
task_dirs = [i for i in basedir.glob(
             '*/secondlevel-RT-True_beta-False_maps')]
for current_task_dir in task_dirs:
    taskname = current_task_dir.parts[-2]
    print('-' * 20)
    print(taskname)
    print('-' * 20)
    get_regressor_correlations(current_task_dir)
    print(" ")
    count_analysis_dirs_and_reaction_time_contrasts(current_task_dir)
    search_analysis_make_figures(current_task_dir)
# -


