# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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
                plot_stat_map(thresh_tstat,
                              title=f"{taskname}:{dependent_variable_name} \
                              correlated with {independent_variable_name}",
                              display_mode='z', threshold=0, cut_coords=10,
                              black_bg=True)
            else:
                print(f"{dependent_variable_name} has no significant \
                correlation with {independent_variable_name}")


# +
def main():
    basedir = Path('/Users/jeanettemumford/sherlock_local/uh2/aim1/BIDS_scans/\
    derivatives/2ndlevel_4_2_21')
    task_dirs = [i for i in basedir.glob(
                 '*/secondlevel-RT-True_beta-False_maps')]

    for current_task_dir in task_dirs:
        taskname = current_task_dir.parts[-2]
        print('-' * 20)
        print(taskname)
        print('-' * 20)
        search_analysis_make_figures(current_task_dir)


if __name__ == '__main__':
    main()
