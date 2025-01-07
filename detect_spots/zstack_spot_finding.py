# 3D spot finding
from skimage import feature, io, exposure, draw
import numpy as np
from tifffile import imwrite
import pandas as pd
import random
from sklearn.cluster import KMeans
from skimage import filters
import sys

def save_blobs_on_movie(blobs_df, movie, file_name):
    # on each slice, label blobs: combine as one movie
    if ("location" in blobs_df.columns):
        labels_nucl = np.zeros_like(movie)
        labels_cyt = np.zeros_like(movie)
        for row in blobs_df.iterrows():
            if(row[1].location == 'nuclear'):
                labels=labels_nucl
            elif(row[1].location == 'cytoplasmic'):
                labels=labels_cyt

            # draw circle on image
            rr, cc = draw.circle_perimeter(int(row[1]['row (y)']), int(row[1]['col (x)']), int(row[1]['radius']),
                                           shape=labels[0].shape)
            labels[int(row[1]['plane (z)']), rr, cc] = 65535
        ij_stack = np.stack([movie, labels_nucl, labels_cyt], axis=1)
    else:
        labels = np.zeros_like(movie)
        for row in blobs_df.iterrows():
            # draw circle on image
            rr, cc = draw.circle_perimeter(int(row[1]['row (y)']), int(row[1]['col (x)']), int(row[1]['radius']),
                                           shape=labels[0].shape)
            labels[int(row[1]['plane (z)']), rr, cc] = 65535
        ij_stack = np.stack([movie, labels], axis=1)

    imwrite(file_name, ij_stack, imagej=True, metadata={'axes': 'ZCYX'})

def get_roi_coords(rois, img_shape):

    roi_to_coords = {}

    for key in rois.keys():
        unknown_roi = False
        roi = rois[key]

        if (roi['type'] == 'polygon' or
                (roi['type'] == 'freehand' and 'x' in roi and 'y' in roi) or
                (roi['type'] == 'traced' and 'x' in roi and 'y' in roi)):

            col_coords = roi['x']
            row_coords = roi['y']
            rr, cc = draw.polygon(row_coords, col_coords, shape=img_shape)

        elif (roi['type'] == 'rectangle'):

            rr, cc = draw.rectangle((roi['top'], roi['left']),
                                    extent=(roi['height'], roi['width']),
                                    shape=img_shape)
            rr = rr.astype('int')
            cc = cc.astype('int')

        elif (roi['type'] == 'oval'):

            rr, cc = draw.ellipse(roi['top'] + roi['height'] / 2,
                                  roi['left'] + roi['width'] / 2,
                                  roi['height'] / 2,
                                  roi['width'] / 2,
                                  shape=img_shape)
        else:
            unknown_roi = True

        if (not unknown_roi):
            roi_to_coords[key] = list(zip(rr,cc))

    return roi_to_coords


def make_mask_from_rois(rois, img_shape):
    final_img = np.zeros(img_shape, dtype='uint8')
    label = 1
    label_to_roi = {}

    for key in rois.keys():
        unknown_roi = False
        roi = rois[key]

        if (roi['type'] == 'polygon' or
                (roi['type'] == 'freehand' and 'x' in roi and 'y' in roi) or
                (roi['type'] == 'traced' and 'x' in roi and 'y' in roi)):

            col_coords = roi['x']
            row_coords = roi['y']
            rr, cc = draw.polygon(row_coords, col_coords, shape=img_shape)

        elif (roi['type'] == 'rectangle'):

            rr, cc = draw.rectangle((roi['top'], roi['left']),
                                    extent=(roi['height'], roi['width']),
                                    shape=img_shape)
            rr = rr.astype('int')
            cc = cc.astype('int')

        elif (roi['type'] == 'oval'):

            rr, cc = draw.ellipse(roi['top'] + roi['height'] / 2,
                                  roi['left'] + roi['width'] / 2,
                                  roi['height'] / 2,
                                  roi['width'] / 2,
                                  shape=img_shape)
        else:
            unknown_roi = True

        if (not unknown_roi):
            final_img[rr, cc] = label
            label_to_roi[label] = key
            label += 1

    return final_img, label_to_roi


def find_spots(spot_stack, nuclei_stack, intensity_stacks, intensity_ch_names,
               rois, labels_file_name='',
               blob_th=0.02, blob_th_rel=None,
               blob_min_s=1, blob_max_s=3):

    # (1) Detect blobs
    spot_stack_rescl = exposure.rescale_intensity(spot_stack)

    print("Detecting blobs...")
    blobs = feature.blob_log(spot_stack_rescl, min_sigma=blob_min_s, max_sigma=blob_max_s,
                             num_sigma=(blob_max_s-blob_min_s+1),
                             threshold=blob_th, threshold_rel=blob_th_rel, overlap=0.5)

    print("Finished...")

    # (2) go through blobs, label with the ROI, get the intensity of nuclei channel and the other channels
    roi_mask, labels_dict = make_mask_from_rois(rois, spot_stack[0].shape)
    if(labels_file_name):
        io.imsave(labels_file_name, roi_mask)

    blobs = blobs[np.argsort(blobs[:, 0])].copy()
    blobs_arr=[]
    for p, r, c, sigma in blobs:
        radius = 2 * np.sqrt(sigma)
        p=int(p)
        r=int(r)
        c=int(c)

        # get 2d blob coordinates (as a disk)
        rr, cc = draw.disk((r, c), int(radius), shape=spot_stack[0].shape)

        # get mean intensity for these coordinates at the specified z-level
        # simplification here since only taking mean intensity at one z-level
        mean_intens_arr = []
        mean_intens_arr.append(np.mean(nuclei_stack[p][rr, cc]))
        mean_intens_arr.append(np.mean(spot_stack[p][rr, cc]))

        for ch_stack in intensity_stacks:
            mean_intens_arr.append(np.mean(ch_stack[p][rr, cc]))

        # which ROI?
        label = roi_mask[int(r)][int(c)]
        if (label > 0):
            roi_key = labels_dict[label]
        else:
            roi_key = ''

        data_arr = [p, r, c, radius, label, roi_key]
        data_arr.extend(mean_intens_arr)
        blobs_arr.append(data_arr)

    cols = ['plane (z)', 'row (y)', 'col (x)', 'radius', 'label', 'roi', 'nuclei_ch_intensity', 'spot_ch_intensity',]
    for name in intensity_ch_names:
        cols.append(f"coloc_ch{name}_intensity")
    return pd.DataFrame(blobs_arr, columns=cols)


def place_spots(coords, n, spot_r, z_dist, type, th, nuclei_stack, seed):
    # Places random spots with given numbers specific to cyto or nucl based on threshold

    random_spots_dict = {}
    coords_choice = coords.copy()
    num_spots_placed=0

    random.seed(seed)
    seeds = random.sample(range(sys.maxsize),(n*2*1000))
    seed_i=0
    while(num_spots_placed < n):

        random.seed(seeds[seed_i])
        seed_i+=1
        cur_coords = random.choice(coords_choice)
        r=int(cur_coords[0])
        c=int(cur_coords[1])

        # select z-channel for spot
        random.seed(seeds[seed_i])
        seed_i += 1
        p = int(random.choice(z_dist.to_numpy()))

        if((p, r, c) in random_spots_dict):
            continue

        place_spot=False
        if(type != "nuclear" and type != "cytoplasmic"):
            place_spot=True
        else:
            # check intensity for this location
            rr, cc = draw.disk((r, c), int(spot_r), shape=nuclei_stack[0].shape)

            # get mean intensity for these coordinates at the specified z-level
            spot_intensity = np.mean(nuclei_stack[p][rr, cc])

            if ((type == 'nuclear' and spot_intensity > th) or (type == 'cytoplasmic' and spot_intensity <= th)):
                place_spot=True

        if(place_spot):
            # place spot
            random_spots_dict[(p, r, c)] = 1
            coords_choice.remove(cur_coords)
            num_spots_placed+=1

    return random_spots_dict.keys()


def randomize_spots(spot_stack, nuclei_stack, intensity_stacks, intensity_ch_names,
                    rois, real_spot_df, seed=0):
    # read in the roi and get the list of roi coordinates.
    # draw X number of random spots on the roi region, where X is same as detected spot count for each roi
    # spot radius will be uniform: the 'size' of the spots as quantified by blob_log is almost always the same
    # get the intensity of nuclei and RNA channels for each spot and save to file

    blobs_arr = []

    coords_dict = get_roi_coords(rois, nuclei_stack[0].shape)
    random.seed(seed)
    roi_seeds = random.sample(range(sys.maxsize), len(coords_dict.keys()))
    seed_i = 0
    for roi_i,roi in enumerate(coords_dict.keys()):
        roi_coords = coords_dict[roi]

        num_spots = real_spot_df[real_spot_df.roi==roi].shape[0]
        if (num_spots > 0):

            radius = real_spot_df[real_spot_df.roi == roi]['radius'].mean()
            z_ch_dist = real_spot_df[real_spot_df.roi == roi]['plane (z)']

            spot_positions = place_spots(roi_coords, num_spots, radius, z_ch_dist, "", 0, nuclei_stack,
                                         roi_seeds[seed_i])
            seed_i += 1
            for (p, r, c) in spot_positions:
                # get 2d blob coordinates (as a disk)
                rr, cc = draw.disk((r, c), int(radius), shape=nuclei_stack[0].shape)

                # get mean intensity for these coordinates at the specified z-level
                mean_intens_arr = []
                mean_intens_arr.append(np.mean(nuclei_stack[p][rr, cc]))
                mean_intens_arr.append(np.mean(spot_stack[p][rr, cc]))

                for ch_stack in intensity_stacks:
                    mean_intens_arr.append(np.mean(ch_stack[p][rr, cc]))

                data_arr = [p, r, c, radius, roi]
                data_arr.extend(mean_intens_arr)
                blobs_arr.append(data_arr)

    cols = ['plane (z)', 'row (y)', 'col (x)', 'radius', 'roi', 'nuclei_ch_intensity', 'spot_ch_intensity', ]
    for name in intensity_ch_names:
        cols.append(f"coloc_ch{name}_intensity")
    return pd.DataFrame(blobs_arr, columns=cols)


def randomize_spots_with_loc(spot_stack, nuclei_stack, intensity_stacks, intensity_ch_names,
                             rois, real_spot_df, loc_th_df, seed=0):
    # read in the roi and get the list of roi coordinates.
    # draw X number of random spots on the roi region, where X is same as detected spot count for each roi
    # spot radius will be uniform: the 'size' of the spots as quantified by blob_log is almost always the same
    # get the intensity of nuclei, spot, and coloc channels for each spot and save to file

    blobs_arr = []

    coords_dict = get_roi_coords(rois, nuclei_stack[0].shape)
    random.seed(seed)
    roi_seeds = random.sample(range(sys.maxsize), len(coords_dict.keys())*2)
    seed_i=0
    for roi in coords_dict.keys():
        roi_coords = coords_dict[roi]

        for loc in ['cytoplasmic','nuclear']:
            cur_spot_df = real_spot_df[(real_spot_df.roi==roi) & (real_spot_df.location==loc)]
            num_spots = cur_spot_df.shape[0]
            if(num_spots > 0):

                cutoff = loc_th_df[loc_th_df['roi']==roi].iloc[0]['th']
                radius = cur_spot_df[cur_spot_df.roi == roi]['radius'].mean()
                z_ch_dist = cur_spot_df[cur_spot_df.roi == roi]['plane (z)']

                spot_positions = place_spots(roi_coords, num_spots, radius, z_ch_dist, loc, cutoff, nuclei_stack,
                                             roi_seeds[seed_i])
                seed_i+=1
                for (p,r,c) in spot_positions:
                    rr, cc = draw.disk((r, c), int(radius), shape=nuclei_stack[0].shape)

                    # get mean intensity for these coordinates at the specified z-level
                    mean_intens_arr = []
                    mean_intens_arr.append(np.mean(nuclei_stack[p][rr, cc]))
                    mean_intens_arr.append(np.mean(spot_stack[p][rr, cc]))

                    for ch_stack in intensity_stacks:
                        mean_intens_arr.append(np.mean(ch_stack[p][rr, cc]))

                    data_arr = [p, r, c, radius, loc, roi]
                    data_arr.extend(mean_intens_arr)
                    blobs_arr.append(data_arr)

    cols = ['plane (z)', 'row (y)', 'col (x)', 'radius', 'location', 'roi', 'nuclei_ch_intensity', 'spot_ch_intensity', ]
    for name in intensity_ch_names:
        cols.append(f"coloc_ch{name}_intensity")
    return pd.DataFrame(blobs_arr, columns=cols)


def locate_spots(spot_df):
    """
    Locate spots in the given spot dataframe to nucleus or cytoplasm based on nucleus channel intensity.

    Parameters:
    spot_df (DataFrame): The spot dataframe containing spot information.

    Returns:
    tuple: A tuple containing the updated spot dataframe and an output dataframe containing counts in nucleus and cytoplasm.
    """
    spot_df['location']=''
    output_arr = []
    for roi in spot_df.roi.unique():
        data = spot_df[spot_df.roi == roi]['nuclei_ch_intensity']

        if (len(data) > 1):
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(data.to_numpy().reshape(-1, 1))

            if (data[kmeans.labels_ == 0].max() < data[kmeans.labels_ == 1].min()):
                th1 = (data[kmeans.labels_ == 0].max() + data[kmeans.labels_ == 1].min()) / 2
            else:
                th1 = (data[kmeans.labels_ == 1].max() + data[kmeans.labels_ == 0].min()) / 2

            th2 = filters.threshold_otsu(data.to_numpy())

            final_th = (th1+th2)/2

            spot_df.loc[(spot_df.roi == roi) & (spot_df.nuclei_ch_intensity > final_th), 'location'] = 'nuclear'
            spot_df.loc[(spot_df.roi == roi) & (spot_df.nuclei_ch_intensity <= final_th), 'location'] = 'cytoplasmic'

            output_arr.append([roi, len(data), final_th, len(data[data > final_th]), len(data[data <= final_th])])
        else:
            output_arr.append([roi, len(data), 0, 0, 0])

    output_df = pd.DataFrame(output_arr, columns=["roi", "num_spots", "th", "num_nuclei_spots", "num_cyto_spots"])

    return spot_df, output_df








