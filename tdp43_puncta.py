import os
import pandas as pd
import glob
from read_roi import read_roi_zip, read_roi_file
from detect_spots import zstack_spot_finding
from skimage import io
import re


home_dir = f"{os.path.expanduser('~')}/Dropbox (NYU Langone Health)"
img_dir = "For Sarah from Martina/TDP-43 puncta soma"

input_dir = f"{home_dir}/{img_dir}"
output_dir = f"{input_dir}/results"
