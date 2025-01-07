import os
import pandas as pd
import glob
from read_roi import read_roi_zip, read_roi_file
from detect_spots import zstack_spot_finding
from skimage import io

# /Users/sarahkeegan/NYU Langone Health Dropbox/Sarah Keegan/mac_files/holtlab/data_and_results/LINE1/Farida-L1-RNAFISH/TIF-roi
home_dir = f"{os.path.expanduser('~')}/NYU Langone Health Dropbox/Sarah Keegan/mac_files"
img_dir = "holtlab/data_and_results/LINE1/Farida-L1-RNAFISH"

input_dir = f"{home_dir}/{img_dir}/TIF-roi"
output_dir = f"{input_dir}/results"

roi_suffix = "_rois"

spot_type = "ORF1"  # "ORF1" #"RNA"
nuclei_channel = 0

# Set this to True to set a single th and min/max sigma for all images
# Set these values once for RNA and once for ORF1
batch_mode = True   # True

if spot_type == "RNA":

    # for RNA spots:
    spot_channel = 2
    intensity_channels = [1, ]
    my_seed = 0

    th = {'SS0016_withdox-1_001': 0.02, 'SS0016_withdox-1_002': 0.02, 'SS0016_withdox-1_003': 0.02,
          'SS0016_withdox-1_004': 0.005, 'SS0016_withdox-1_005': 0.005}
    th2 = None
    min_sigma = {'SS0016_withdox-1_001': 1, 'SS0016_withdox-1_002': 1, 'SS0016_withdox-1_003': 1,
                 'SS0016_withdox-1_004': 1, 'SS0016_withdox-1_005': 1}
    max_sigma = {'SS0016_withdox-1_001': 3, 'SS0016_withdox-1_002': 3, 'SS0016_withdox-1_003': 3,
                 'SS0016_withdox-1_004': 3, 'SS0016_withdox-1_005': 3}

    batch_th = 0.005
    batch_min_sigma = 1
    batch_max_sigma = 3

elif spot_type == "ORF1":

    # for ORF1 spots:
    spot_channel = 1
    intensity_channels = [2, ]
    my_seed = 1

    th = {'SS0016_withdox-1_001': 0.025, 'SS0016_withdox-1_002': 0.03, 'SS0016_withdox-1_003': 0.075,
          'SS0016_withdox-1_004': 0.08, 'SS0016_withdox-1_005': 0.075}
    th2 = None
    min_sigma = {'SS0016_withdox-1_001': 1, 'SS0016_withdox-1_002': 2, 'SS0016_withdox-1_003': 1,
                 'SS0016_withdox-1_004': 1, 'SS0016_withdox-1_005': 1}
    max_sigma = {'SS0016_withdox-1_001': 3, 'SS0016_withdox-1_002': 3, 'SS0016_withdox-1_003': 3,
                 'SS0016_withdox-1_004': 2, 'SS0016_withdox-1_005': 3}

    batch_th = 0.005
    batch_min_sigma = 1
    batch_max_sigma = 3

else:
    raise Exception("Error: invalid spot_type.")


movie_files = glob.glob(f"{input_dir}/*.tif")

full_df = pd.DataFrame()
full_random_df = pd.DataFrame()
full_loc_counts_df = pd.DataFrame()

for movie_file in movie_files:
    file_root = os.path.splitext(os.path.split(movie_file)[1])[0]
    if (os.path.exists(f"{input_dir}/{file_root}{roi_suffix}.zip")):
        rois = read_roi_zip(f"{input_dir}/{file_root}{roi_suffix}.zip")
    else:
        rois = read_roi_file(f"{input_dir}/{file_root}{roi_suffix}.roi")
    full_stack = io.imread(f"{movie_file}")

    print(movie_file)
    #if(file_root != 'zstack_005'):
    #    continue

    # Save blobs on movie, color indicates location, if using
    # If image shape is (z, x, y, c)
    spot_stack = full_stack[:, :, :, spot_channel]
    nuclei_stack = full_stack[:, :, :, nuclei_channel]

    # If image shape is (z, c, x, y)
    # spot_stack = full_stack[:, spot_channel, :, :]
    # nuclei_stack = full_stack[:, nuclei_channel, :, :]

    intensity_stacks=[]
    ch_names=[]
    for ch in intensity_channels:
        ch_names.append(str(ch))

        # Fix for image shape (z, x, y, c) or (z, c, x, y)
        intensity_stacks.append(full_stack[:, :, :, ch])
        # intensity_stacks.append(full_stack[:, ch, :, :])

    # find spots
    if batch_mode:
        blob_th = batch_th
        blob_min_s = batch_min_sigma
        blob_max_s = batch_max_sigma
    else:
        blob_th = th[file_root]
        blob_min_s = min_sigma[file_root]
        blob_max_s = max_sigma[file_root]

    blobs_df = zstack_spot_finding.find_spots(spot_stack, nuclei_stack, intensity_stacks, ch_names,
                                              rois, labels_file_name=f"{output_dir}/{file_root}-roi_labels.tif",
                                              blob_th=blob_th, blob_th_rel=th2,
                                              blob_min_s=blob_min_s, blob_max_s=blob_max_s)

    # use nucleus signal to determine spot location (cytoplasmic or nuclear)
    # this adds a few columns to the data frame indicating the threshold levels and the location
    blobs_df, loc_counts_df = zstack_spot_finding.locate_spots(blobs_df)

    # save blobs on movie for checking - if location column exists in data frame, they will be colored by location
    zstack_spot_finding.save_blobs_on_movie(blobs_df, spot_stack,
                                            file_name=f"{output_dir}/{file_root}-{spot_type}-marked_blobs.tif")
    zstack_spot_finding.save_blobs_on_movie(blobs_df, nuclei_stack,
                                            file_name=f"{output_dir}/{file_root}-{spot_type}-marked_blobs-nucl.tif")

    # randomize spots
    random_blobs_df = zstack_spot_finding.randomize_spots_with_loc(spot_stack, nuclei_stack, intensity_stacks, ch_names,
                                                                   rois, blobs_df, loc_counts_df, seed=my_seed)

    # save random blobs on movie for checking - location only
    zstack_spot_finding.save_blobs_on_movie(random_blobs_df, nuclei_stack,
                                            file_name=f"{output_dir}/{file_root}-{spot_type}-marked_random_blobs-nucl.tif")

    loc_counts_df['file_name'] = file_root
    random_blobs_df['file_name'] = file_root

    full_random_df = pd.concat([full_random_df, random_blobs_df], axis=0, ignore_index=True)
    full_loc_counts_df = pd.concat([full_loc_counts_df, loc_counts_df], axis=0, ignore_index=True)

    blobs_df['file_name'] = file_root
    full_df = pd.concat([full_df, blobs_df], axis=0, ignore_index=True)

# Filter spots not within an ROI
full_df=full_df[full_df['roi'] != '']
full_df.index = range(len(full_df))

full_random_df=full_random_df[full_random_df['roi'] != '']
full_random_df.index = range(len(full_random_df))

full_loc_counts_df=full_loc_counts_df[full_loc_counts_df['roi'] != '']
full_loc_counts_df.index = range(len(full_loc_counts_df))

# Save final output files
full_df.to_csv(f"{output_dir}/{spot_type}-all_spots.txt", sep='\t')
full_random_df.to_csv(f"{output_dir}/{spot_type}-all_random_spots.txt", sep='\t')
full_loc_counts_df.to_csv(f"{output_dir}/{spot_type}-spot_counts.txt", sep='\t')

