import os
import pandas as pd
import glob
from read_roi import read_roi_zip
from detect_spots import zstack_spot_finding
from skimage import io


home_dir = f"{os.path.expanduser('~')}/Dropbox (NYU Langone Health)/mac_files"
img_dir = "holtlab/data_and_results/LINE1/RNA-ORF1 Colocalization/spot_counting/20221030_LINE1/"

input_dir = f"{home_dir}/{img_dir}/dox"
output_dir = f"{input_dir}/results"

roi_suffix = "RoiSet"

spot_type = "RNA" #"ORF1" #"RNA"
nuclei_channel = 1

if(spot_type == "RNA"):

    # for RNA (ch 0) spots:
    spot_channel = 0
    intensity_channels = [3,]

    th={'zstack_001': 0.02, 'zstack_002': 0.02, 'zstack_003': 0.02, 'zstack_004': 0.02, 'zstack_005': 0.005}
    th2 = None
    min_sigma = {'zstack_001': 1, 'zstack_002': 1, 'zstack_003': 1, 'zstack_004': 1, 'zstack_005': 1}
    max_sigma = {'zstack_001': 3, 'zstack_002': 3, 'zstack_003': 3, 'zstack_004': 3, 'zstack_005': 3}

elif(spot_type == "ORF1"):

    # for ORF1 (ch 3) spots:
    spot_channel = 3
    intensity_channels = [0,]

    th = {'zstack_001': 0.025, 'zstack_002': 0.03, 'zstack_003': 0.075, 'zstack_004': 0.08, 'zstack_005': 0.075}
    th2 = None
    min_sigma = {'zstack_001': 1, 'zstack_002': 2, 'zstack_003': 1, 'zstack_004': 1, 'zstack_005': 1}
    max_sigma = {'zstack_001': 3, 'zstack_002': 3, 'zstack_003': 3, 'zstack_004': 2, 'zstack_005': 3}

else:
    raise Exception("Error: invalid spot_type.")


movie_files = glob.glob(f"{input_dir}/*.tif")

full_df = pd.DataFrame()
full_random_df = pd.DataFrame()
full_loc_counts_df = pd.DataFrame()

for movie_file in movie_files:
    file_root = os.path.splitext(os.path.split(movie_file)[1])[0]
    rois = read_roi_zip(f"{input_dir}/{file_root}_{roi_suffix}.zip")
    full_stack = io.imread(f"{movie_file}")

    print(movie_file)
    #if(file_root != 'zstack_005'):
    #    continue

    # Save blobs on movie, color indicates location, if using
    spot_stack = full_stack[:, spot_channel, :, :]
    nuclei_stack = full_stack[:, nuclei_channel, :, :]

    # find spots
    blobs_df = zstack_spot_finding.find_spots(full_stack, rois,
                                              labels_file_name=f"{output_dir}/{file_root}-roi_labels.tif",
                                              spot_ch=spot_channel, nucl_ch=nuclei_channel, intens_chs=intensity_channels,
                                              blob_th=th[file_root], blob_th_rel=th2,
                                              blob_min_s=min_sigma[file_root], blob_max_s=max_sigma[file_root])

    # use nucleus signal to determine spot location (cytoplasmic or nuclear)
    # this adds a few columns to the data frame indicating the threshold levels and the location
    blobs_df, loc_counts_df = zstack_spot_finding.locate_spots(blobs_df)

    # save blobs on movie for checking - if location column exists in data frame, they will be colored by location
    zstack_spot_finding.save_blobs_on_movie(blobs_df, spot_stack,
                                            file_name=f"{output_dir}/{file_root}-{spot_type}-marked_blobs.tif")
    zstack_spot_finding.save_blobs_on_movie(blobs_df, nuclei_stack,
                                            file_name=f"{output_dir}/{file_root}-{spot_type}-marked_blobs-nucl.tif")

    # randomize spots
    random_blobs_df = zstack_spot_finding.randomize_spots_with_loc(full_stack, rois, blobs_df, loc_counts_df,
                                                          spot_ch=spot_channel,
                                                          nucl_ch=nuclei_channel,
                                                          intens_chs=intensity_channels)

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