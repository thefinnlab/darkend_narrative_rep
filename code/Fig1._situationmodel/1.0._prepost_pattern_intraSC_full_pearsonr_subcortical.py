"""
This script calculates pattern intra-SC values between listens for each ROI per subject at each TR
It saves the results for each node as a NumPy file that is then used in the next script. 

Dependencies:
- Python 3.7+
- Nilearn, NumPy, SciPy, Statsmodels, Brainiak, and Seaborn libraries.
"""

import os
import re
import sys
import numpy as np
from scipy.stats import pearsonr
from glob import glob




schaeffer_dir = '../../data/_subcortical_rois/'
data_dir = '/dartfs/rc/lab/F/FinnLab/darkend-fmri/allfuncs'

func_filenames_part1 = glob(os.path.join(data_dir, ('sub-*_func_task-darkend1*.nii.gz')))
func_filenames_part1 = sorted(func_filenames_part1)
func_filenames_part2 = glob(os.path.join(data_dir, ('sub-*_func_task-darkend2*.nii.gz')))
func_filenames_part2 = sorted(func_filenames_part2)


sub_nums = [re.search('sub-(\d+)', fname).group(1) for fname in func_filenames_part1]
all_subs_npy = [f'sub-{sub}' for sub in sub_nums]




def grab_TRs(sub_idx,listen,node):
    """
    Extracts the neural time series data for a specific subject, listen condition, and node.

    Parameters:
    - sub_idx (int): Subject index.
    - listen (int): Listen condition (1 or 2).
    - node (int): Node index.

    Returns:
    - np.ndarray: Neural data (TRs x voxels).
    """
    node_array = np.load(schaeffer_dir + f'_part{listen}_all_subs_node_{node}_darkend_7N.npy',allow_pickle=True)
    node_array = np.moveaxis(node_array,2,0) #subject by TR by voxels! 
    this_sub = node_array[sub_idx,:,:]
    return this_sub

def calculate_correlation(A, B):
    """
    Computes the Pearson correlation between two time series, ignoring NaN values.

    Parameters:
    - A (np.ndarray): Time series data for Listen 1.
    - B (np.ndarray): Time series data for Listen 2.

    Returns:
    - float: Pearson correlation coefficient.
    """
    ## Create a mask where both A and B are not NaN
    mask = ~np.isnan(A) & ~np.isnan(B)

    # Use the mask to filter A and B
    A_masked = A[mask]
    B_masked = B[mask]
    
    # Calculate the pearson correlation 
    correlation, _ = pearsonr(A_masked, B_masked)
    
    return correlation


all_TRs = list(range(1118)) 
#this is the range of the full preprocessed data including the 2 TRs added to the start of the stimulus and 10 TRs added to the end of the stimulus 
#we crop off time to only fit the stimulus when comparing the pre and post-twist

nodes = int(sys.argv[1])


sub_corr_reint = {}

for node in [nodes]:
    print("Doing node {} of {}...".format(node, 100), end =" ")

    
    for sub in range(36):
        sub_brain_L1 = grab_TRs(sub,1,node) #setting the Listen
        sub_brain_L2 = grab_TRs(sub,2,node)
        
        corr_vals = []
        for TR in all_TRs:

            #Computing it at each TR! 
            time_L1 = sub_brain_L1[TR:TR+1,:]
            time_L2 = sub_brain_L2[TR:TR+1,:]
            
            time_L1_avg = np.mean(time_L1,axis=0) #this is now going to be the size of the voxels
            time_L2_avg = np.mean(time_L2,axis=0)        

            time_L1_avg = time_L1_avg[np.newaxis,:] 
            time_L2_avg = time_L2_avg[np.newaxis,:]
            corr_vals.append(calculate_correlation(time_L1_avg,time_L2_avg))

        sub_corr_reint[sub] = corr_vals
    
    np.save(f'../../data/_multivariate_intraSC/intraSC_pattern_all_node{node}_pearsonr.npy',sub_corr_reint)

    


