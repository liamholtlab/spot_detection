from skimage import feature, io, exposure, draw
import numpy as np
from tifffile import imwrite
import pandas as pd
import random
from sklearn.cluster import KMeans
from skimage import filters
import sys


def find_spots(spot_img, blob_th=0.02, blob_th_rel=None, blob_min_s=1, blob_max_s=3):

    # (1) Detect blobs
    spot_img_rescl = exposure.rescale_intensity(spot_img)

    blobs = feature.blob_log(spot_img_rescl, min_sigma=blob_min_s, max_sigma=blob_max_s,
                             num_sigma=(blob_max_s-blob_min_s+1),
                             threshold=blob_th, threshold_rel=blob_th_rel, overlap=0.5)

    blobs = blobs[np.argsort(blobs[:, 0])].copy()
    blobs_arr=[]
    for r, c, sigma in blobs:
        radius = 2 * np.sqrt(sigma)
        r = int(r)
        c = int(c)

        # get 2d blob coordinates (as a disk)
        rr, cc = draw.disk((r, c), int(radius), shape=spot_img.shape)
        data_arr = [r, c, radius, np.mean(spot_img[rr, cc])]
        blobs_arr.append(data_arr)

    cols = ['row (y)', 'col (x)', 'radius', 'label', 'spot_intensity']
    return pd.DataFrame(blobs_arr, columns=cols)