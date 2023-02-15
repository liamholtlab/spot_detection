import os
import pandas as pd
import glob
from read_roi import read_roi_zip, read_roi_file
from detect_spots import zstack_spot_finding
from skimage import io
import numpy as np

home_dir = f"{os.path.expanduser('~')}/Dropbox (NYU Langone Health)/mac_files"
img_dir = "holtlab/data_and_results/LINE1/ORF1-ORF1 Colocalization"

input_dir = f"{home_dir}/{img_dir}"
output_dir = f"{input_dir}/results"

roi_suffix = ""

spot_type = "ORF1-561"  # only one spot detection here
nuclei_channel = 0

if (spot_type == "ORF1-561"):

    spot_channel = 2
    intensity_channels = [1,3]

    th = 0.04
    th2 = None
    min_sigma = 1
    max_sigma = 3

else:
    raise Exception("Error: invalid spot_type.")

# Spot detection plus measure intensity in 2 channels:
movie_files = glob.glob(f"{input_dir}/*.tif")

full_df = pd.DataFrame()
full_random_df = pd.DataFrame()
full_loc_counts_df = pd.DataFrame()

for movie_file in movie_files:
    file_root = os.path.splitext(os.path.split(movie_file)[1])[0]

    if(os.path.exists(f"{input_dir}/{file_root}{roi_suffix}.zip")):
        rois = read_roi_zip(f"{input_dir}/{file_root}{roi_suffix}.zip")
    else:
        rois = read_roi_file(f"{input_dir}/{file_root}{roi_suffix}.roi")
    full_stack = io.imread(f"{movie_file}")

    print(movie_file)

    # Save blobs on movie, color indicates location, if using
    spot_stack = full_stack[:, :, :, spot_channel]
    nuclei_stack = full_stack[:, :, :, nuclei_channel]

    intensity_stacks = []
    ch_names = []
    for ch in intensity_channels:
        ch_names.append(str(ch))
        intensity_stacks.append(full_stack[:, :, :, ch])

    # find spots
    blobs_df = zstack_spot_finding.find_spots(spot_stack, nuclei_stack, intensity_stacks, ch_names,
                                              rois, labels_file_name=f"{output_dir}/{file_root}-roi_labels.tif",
                                              blob_th=th, blob_th_rel=th2,
                                              blob_min_s=min_sigma, blob_max_s=max_sigma)

    # save blobs on movie for checking
    zstack_spot_finding.save_blobs_on_movie(blobs_df, spot_stack,
                                            file_name=f"{output_dir}/{file_root}-{spot_type}-marked_blobs.tif")

    # randomize spots
    random_blobs_df = zstack_spot_finding.randomize_spots(spot_stack, nuclei_stack, intensity_stacks, ch_names,
                                                          rois, blobs_df, seed=0)

    # save random blobs on movie for checking
    zstack_spot_finding.save_blobs_on_movie(random_blobs_df, spot_stack,
                                            file_name=f"{output_dir}/{file_root}-{spot_type}-marked_random_blobs.tif")

    random_blobs_df['file_name'] = file_root
    full_random_df = pd.concat([full_random_df, random_blobs_df], axis=0, ignore_index=True)

    blobs_df['file_name'] = file_root
    full_df = pd.concat([full_df, blobs_df], axis=0, ignore_index=True)

# Filter spots not within an ROI
full_df = full_df[full_df['roi'] != '']
full_df.index = range(len(full_df))

full_random_df = full_random_df[full_random_df['roi'] != '']
full_random_df.index = range(len(full_random_df))

# Save final output files
full_df.to_csv(f"{output_dir}/{spot_type}-all_spots.txt", sep='\t')
full_random_df.to_csv(f"{output_dir}/{spot_type}-all_random_spots.txt", sep='\t')

