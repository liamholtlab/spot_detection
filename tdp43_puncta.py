import os
import pandas as pd
import glob
from detect_spots import spot_finding
from skimage import io
import re
from nd2reader import ND2Reader


home_dir = f"{os.path.expanduser('~')}/Dropbox (NYU Langone Health)"

cell_types = ["neurites", "soma"]
file_ext = {"soma": "tif", "neurites": "nd2"}

# load a file, apply spot detection to each frame.
# overwrite the labeled spots on the image
# save to results

for cell_type in cell_types:
    img_dir = f"{home_dir}/For Sarah from Martina/TDP-43 puncta {cell_type}"
    output_dir = f"{img_dir}/results"

    ext = file_ext[cell_type]
    movie_files = glob.glob(f"{img_dir}/*.{ext}")
    for movie_file in movie_files:
        if ext == 'nd2':
            images = ND2Reader(movie_file)
            images.bundle_axes = 'tyx'
            images = images[0]  # ND2Reader adds an extra dimension to the beginning
        else:
            # TIF file contains single channel, each frame is a time point
            images = io.imread(movie_file)

        n_timepoints = len(images)
        print("n_timepoints:", n_timepoints)

        for tp in n_timepoints:

            spot_df = spot_finding.find_spots(images[tp])



