import slideio
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dfply import *

from sklearn.decomposition import NMF
import skimage.measure

from tqdm import tqdm


EPSILON = 1e-5
STAIN_CNT = 5
BG_STAIN = 0
BRAIN_STAIN = 2
IMMUNE_STAIN = 3
TUMOR_STAIN = 4


def read_slide(src, rect=None, size=None, thumbnail_size=None):
    brain_slide = slideio.open_slide(src, 'SVS')
    scene = brain_slide.get_scene(0)
    img_size = scene.size

    if rect is None:
        x0, y0 = 0, 0
        w, h = img_size[0], img_size[1]
        rect = (x0, y0, w, h)
    else:
        if isinstance(rect[0], float):
            x0, y0 = int(rect[0] * img_size[0]), int(rect[1] * img_size[1])
            w, h = int(rect[2] * img_size[0]), int(rect[3] * img_size[1])
            w, h = min(img_size[0] - x0 - 10, w), min(img_size[1] - y0 - 10, h)
            rect = (x0, y0, w, h)

    if thumbnail_size is None:
        width = min(3000, max(2000, rect[2]))
        thumbnail_size = (width, 0)
    if size is None:
        width = min(9000, max(5000, rect[2]))
        size = (width, 0)
    # print("img_size:", img_size)

    # print("==============\nreading thumbnail...", thumbnail_size)
    thumbnail = scene.read_block(rect, thumbnail_size)
    # print("thumbnail read", thumbnail.shape, "\n==============")

    # print("==============\nreading main image...", size)
    image = scene.read_block(rect, size)
    # print("main image read", image.shape, "\n==============")

    # I am sure about the indices, although they might seem confusing
    h_pix_size = 1e6 * scene.resolution[1] * img_size[1] / image.shape[0]
    w_pix_size = 1e6 * scene.resolution[0] * img_size[0] / image.shape[1]
    
    return {'main': image, 'thumbnail': thumbnail, 'h_pix_size': h_pix_size, 'w_pix_size': w_pix_size}

def color_analysis(image):
    main_img, thumbnail = image['main'], image['thumbnail']

    pixels = thumbnail.reshape((-1, 3)) / 255.
    model = NMF(n_components=STAIN_CNT, init='random', random_state=0)
    model.fit(pixels)
    H = model.components_
    print(H * 255)

    seg2, tumor_mask2, brain_mask2, immune_mask2 = colorbased_seg(main_img, H)

    return H, tumor_mask2, brain_mask2, immune_mask2

def colorbased_seg(main_img, stains, kern_s = 2, kern_m = 3, kern_b = 4):
    pixels = main_img.reshape((-1, 1, 3)) / 255.
    conv_pix = cv2.cvtColor(np.float32(pixels), cv2.COLOR_RGB2Lab)
    conv_stains = cv2.cvtColor(np.float32(stains.reshape((-1, 1, 3))), cv2.COLOR_RGB2Lab)
    color_diffs = np.hstack([np.linalg.norm(conv_pix - stain_col, axis=2).reshape((-1,1)) for stain_col in conv_stains])
    detected_color = np.argmin(color_diffs, axis=1)

    small_kernel = np.ones((kern_s,kern_s),np.uint8)
    med_kernel = np.ones((kern_m,kern_m),np.uint8)
    big_kernel = np.ones((kern_b,kern_b),np.uint8)

    tumor_mask = np.zeros(detected_color.shape)
    tumor_mask[detected_color == TUMOR_STAIN] = 1
    tumor_mask = tumor_mask.reshape(main_img.shape[0], main_img.shape[1])
    tumor_mask = cv2.bilateralFilter(tumor_mask.astype(np.uint8), 9, 5, 5)

    brain_mask = np.zeros(detected_color.shape)
    brain_mask[detected_color == BRAIN_STAIN] = 1
    brain_mask = brain_mask.reshape(main_img.shape[0], main_img.shape[1])
    brain_mask = cv2.erode(brain_mask, small_kernel, iterations = 1)
    brain_mask = cv2.dilate(brain_mask, med_kernel, iterations = 2)
    brain_mask = cv2.erode(brain_mask, small_kernel, iterations = 1)
    brain_mask = cv2.dilate(brain_mask, big_kernel, iterations = 2)
    brain_mask = cv2.bilateralFilter(brain_mask.astype(np.uint8), 9, 5, 5)
    overlaps = cv2.bitwise_and(tumor_mask, brain_mask)
    brain_mask = brain_mask - overlaps

    immune_mask = np.zeros(detected_color.shape)
    immune_mask[detected_color == IMMUNE_STAIN] = 1
    immune_mask = immune_mask.reshape(main_img.shape[0], main_img.shape[1])
    immune_mask = cv2.bilateralFilter(immune_mask.astype(np.uint8), 9, 5, 5)
    for other_mask in [tumor_mask, brain_mask]:
        overlaps = cv2.bitwise_and(other_mask, immune_mask)
        immune_mask = immune_mask - overlaps

    seg = np.zeros(main_img.shape)
    seg[:, :, 0] = tumor_mask
    seg[:, :, 1] = brain_mask
    seg[:, :, 2] = immune_mask

    return seg, tumor_mask, brain_mask, immune_mask

def find_tumors(tumor_mask, CMP_PROPORTION = .75, return_colored_comps = False):
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(tumor_mask)
    print("num_labels:", num_labels)
    
    all_areas = stats[:, 4]

    CMP_CNT = int(CMP_PROPORTION * all_areas.shape[0])
    biggest_comps = (-all_areas).argsort()[1:CMP_CNT]
    all_comps = np.unique(labels_im)
    for comp in all_comps:
        if comp not in biggest_comps:
            labels_im[labels_im == comp] = 0
    ######################
    if return_colored_comps:
        label_hue = np.uint8(179*labels_im/np.max(labels_im))
        blank_ch = 255*np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # cvt to RGB for display
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2RGB)

        # set bg label to black
        labeled_img[label_hue==0] = 0
    else:
        labeled_img = None

    return labeled_img, labels_im, biggest_comps

def find_normal_boundaries(brain_mask):
    edge_img = np.zeros(brain_mask.shape)
    contours, _  = cv2.findContours(brain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flat_boundary_pts = np.vstack([cnt.reshape((-1, 2)) for cnt in contours])
    cv2.drawContours(edge_img, contours, -1, 255, 2)
    
    return flat_boundary_pts, edge_img

def extract_features(src, stains, labels_im, biggest_comps, feature_list, h_pix_size, w_pix_size, brain_boundary_pts, max_cmp_cnt=20):
    all_features = []

    seg = np.zeros((*labels_im.shape, 3))
    seg[:, :, 0] = tumor_mask
    seg[:, :, 1] = brain_mask
    
    selection_size = int(max_cmp_cnt / 4)
    rand_g = np.random.default_rng(1)
    main_cmps = np.arange(selection_size)

    sel_ubound = min(200, len(biggest_comps))
    rest_sel_size = min(sel_ubound, max_cmp_cnt) - selection_size
    small_cmps = rand_g.choice(np.arange(selection_size, sel_ubound), rest_sel_size, replace=False)
    selected_indices = np.concatenate((main_cmps, small_cmps))
    selected_indices.sort()
    
    for comp in tqdm(biggest_comps[selected_indices]):
        ############################################################
        # Loading high power tumor
        pixel_locs = np.where(labels_im == comp)
        min_y, max_y = np.amin(pixel_locs[0]), np.amax(pixel_locs[0])
        min_x, max_x = np.amin(pixel_locs[1]), np.amax(pixel_locs[1])
        h = max_y - min_y
        w = max_x - min_x

        min_y, min_x = int(max(min_y - h/20, 0)), int(max(min_x - w/20, 0))
        max_y, max_x = int(min(max_y + h/20, brain_mask.shape[0])), int(min(max_x + w/20, brain_mask.shape[1]))
        w, h = max_x - min_x, max_y - min_y

        min_y /= labels_im.shape[0]; h /= labels_im.shape[0]
        min_x /= labels_im.shape[1]; w /= labels_im.shape[1]

        curr_tumor = read_slide(src, rect = (min_x, min_y, w, h), size=(4000, 0), thumbnail_size=(500, 0))

        # Segmenting high power tumor
        hp_seg, hp_tumor_mask, hp_brain_mask, hp_immune_mask = colorbased_seg(curr_tumor['main'], stains, kern_s=3, kern_m=4, kern_b=5)
        hp_interface, hp_brain_bound = get_interface(hp_brain_mask, hp_tumor_mask, erode_size=15, dilate_size=10)

        hp_real_interface = cv2.bitwise_and(curr_tumor['main'], curr_tumor['main'], mask = hp_brain_bound)
        ############################################################

        curr_im = np.zeros(labels_im.shape, np.uint8)
        curr_im[labels_im != comp] = 0
        curr_im[labels_im == comp] = 255

        seg[labels_im == comp, 2] = 255
        
        # fig, axs = plt.subplots(2, 2)
        # fig.tight_layout()
        # axs[0, 0].imshow(curr_img['main'])
        # axs[0, 0].set_title('Current Image')
        # axs[0, 1].imshow(seg)
        # axs[0, 1].set_title('Segmented')
        # axs[1, 0].imshow(curr_tumor['main'])
        # axs[1, 0].set_title('Current Tumor Component')
        # axs[1, 1].imshow(hp_seg)
        # axs[1, 1].set_title('High Power Segmented')
        # plt.show()

        seg[labels_im == comp, 2] = 0

        if mask_area(curr_im, 1, 1, None, None) < 25:
            break

        curr_features = calculate_features(hp_seg, feature_list, h_pix_size, w_pix_size, brain_boundary_pts)
        all_features.append(curr_features)

    all_features = pd.DataFrame.from_dict(all_features)

    return all_features

def calculate_features(object_segment, feature_list, h_pix_size, w_pix_size, brain_boundary_pts, return_hidden=False):
    res = {}
    for feature_func in feature_list:
        feature_val = feature_func(object_segment, h_pix_size, w_pix_size, brain_boundary_pts, res)
        if type(feature_val) is dict:
            for key in feature_val:
                res[feature_func.__name__ + '_' + key] = feature_val[key]
        elif type(feature_val) is list:
            for i in range(len(feature_val)):
                res[feature_func.__name__ + '_' + str(i)] = feature_val[i]
        else:
            res[feature_func.__name__] = feature_val
    
    if not return_hidden:
        for key in list(res.keys()):
            if '_' == key[0]: del res[key]
    
    return res

######## Features
def _tiled_switch(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    pool_d = 9
    neighber_d = 3

    kernel = np.ones((neighber_d,neighber_d),np.uint8) / (neighber_d**2 - 1)
    kernel[neighber_d//2, neighber_d//2] = 0

    cont_tile_tumor = skimage.measure.block_reduce(obj_segments[:, :, 0], pool_d, np.mean)
    cont_tile_brain = skimage.measure.block_reduce(obj_segments[:, :, 1], pool_d, np.mean)
    _, bin_tile_tumor = cv2.threshold(cont_tile_tumor, .5, 1, cv2.THRESH_BINARY)
    _, bin_tile_brain = cv2.threshold(cont_tile_brain, .5, 1, cv2.THRESH_BINARY)

    tumor_switch = bin_tile_tumor * cv2.filter2D(bin_tile_brain, -1, kernel)
    brain_switch = bin_tile_brain * cv2.filter2D(bin_tile_tumor, -1, kernel)

    # cont_seg = np.zeros((*cont_tile_tumor.shape, 3))
    # cont_seg[:,:,0] = cont_tile_tumor
    # cont_seg[:,:,1] = cont_tile_brain

    # bin_seg = np.zeros((*bin_tile_tumor.shape, 3))
    # bin_seg[:,:,0] = bin_tile_tumor
    # bin_seg[:,:,1] = bin_tile_brain
    
    switch_seg = np.zeros((*tumor_switch.shape, 3))
    switch_seg[:,:,0] = tumor_switch
    switch_seg[:,:,1] = brain_switch
    
    # fig, axs = plt.subplots(2, 2)
    # fig.tight_layout()
    # axs[0, 0].imshow(obj_segments)
    # axs[0, 0].set_title('Segments')
    # axs[0, 1].imshow(cont_seg)
    # axs[0, 1].set_title('Continuous Tiles')
    # axs[1, 0].imshow(bin_seg)
    # axs[1, 0].set_title('Binary Tiles')
    # axs[1, 1].imshow(switch_seg)
    # axs[1, 1].set_title('Switched Tiles')
    # plt.show()
    
    return switch_seg

def brain_area(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    pix_area = h_pix_size * w_pix_size
    return np.where(obj_segments[:, :, 1] != 0)[0].shape[0] * pix_area

def tumor_area(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    pix_area = h_pix_size * w_pix_size
    return np.where(obj_segments[:, :, 0] != 0)[0].shape[0] * pix_area

def tiled_count(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    pix_area = h_pix_size * w_pix_size
    
    if 'brain_area' in cache:
        brain_area_val = cache['brain_area'] / pix_area
    else:
        brain_area_val = brain_area(obj_segments, 1, 1, brain_boundary_pts, cache)
    
    if 'tumor_area' in cache:
        tumor_area_val = cache['tumor_area'] / pix_area
    else:
        tumor_area_val = tumor_area(obj_segments, 1, 1, brain_boundary_pts, cache)
    
    if '_tiled_switch' in cache:
        tiled_switch = cache['_tiled_switch']
    else:
        tiled_switch = _tiled_switch(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)

    return np.where(tiled_switch != 0)[0].shape[0] / (np.sqrt(brain_area_val) + np.sqrt(tumor_area_val))

def tiled_sum(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    pix_area = h_pix_size * w_pix_size
    
    if 'brain_area' in cache:
        brain_area_val = cache['brain_area'] / pix_area
    else:
        brain_area_val = brain_area(obj_segments, 1, 1, brain_boundary_pts, cache)
    
    if 'tumor_area' in cache:
        tumor_area_val = cache['tumor_area'] / pix_area
    else:
        tumor_area_val = tumor_area(obj_segments, 1, 1, brain_boundary_pts, cache)
    
    if '_tiled_switch' in cache:
        tiled_switch = cache['_tiled_switch']
    else:
        tiled_switch = _tiled_switch(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)

    return np.sum(tiled_switch) / (np.sqrt(brain_area_val) + np.sqrt(tumor_area_val))

def _brain_mask(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    return brain_mask

def _mask_curve(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    contours, _ = cv2.findContours(obj_segments, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def _mask_curvature(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    if '_mask_curve' in cache:
        contours = cache['_mask_curve']
    else:
        contours = _mask_curve(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)

    conversion_mat = np.array([
        [5/4, 1, 3/4],
        [3/2, 0, 1/2],
        [7/4, 0, 1/4]
    ])

    variation = contours[0][1:, 0] - contours[0][:-1, 0]
    variation_ang = conversion_mat[variation[:,0]+1, variation[:,1]+1]
    curvature = variation_ang[1:] - variation_ang[:-1]
    curvature[curvature >= 2] = curvature[curvature >= 2] - 2
    curvature[curvature <= -2] = curvature[curvature <= -2] + 2
    curvature = curvature * np.pi

    return curvature

def _mask_convex_hull(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    if '_mask_curve' in cache:
        contours = cache['_mask_curve']
    else:
        contours = _mask_curve(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    hull = cv2.convexHull(contours[0])

    return hull

def mask_perimeter(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    if '_mask_curve' in cache:
        contours = cache['_mask_curve']
    else:
        contours = _mask_curve(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    
    pix_size = (h_pix_size + w_pix_size) / 2

    return cv2.arcLength(contours[0], True) * pix_size

def mask_area(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    pix_area = h_pix_size * w_pix_size
    return np.where(obj_segments != 0)[0].shape[0] * pix_area

def mask_compactness(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    m_perim = cache['mask_perimeter'] if 'mask_perimeter' in cache else mask_perimeter(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    m_area = cache['mask_area'] if 'mask_area' in cache else mask_area(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    compactness = np.pi * 4 * m_area / m_perim / m_perim

    return compactness

def mask_convexity(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    m_perim = cache['mask_perimeter'] if 'mask_perimeter' in cache else mask_perimeter(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    m_cnx_hull = cache['_mask_convex_hull'] if '_mask_convex_hull' in cache else _mask_convex_hull(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    m_cnx_perim = mask_perimeter(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, {'_mask_curve': [m_cnx_hull]})

    convexity = m_cnx_perim / m_perim

    return convexity

def mask_solidity(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    m_area = cache['mask_area'] if 'mask_area' in cache else mask_area(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    m_cnx_hull = cache['_mask_convex_hull'] if '_mask_convex_hull' in cache else _mask_convex_hull(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    m_cnx_area = cv2.contourArea(m_cnx_hull)

    solidity = m_area / m_cnx_area if m_cnx_area != 0 else 1

    return solidity

def bending_energy(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    if '_mask_curvature' in cache:
        curvature = cache['_mask_curvature']
    else:
        curvature = _mask_curvature(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    
    return np.mean(np.power(curvature, 2))

def total_abs_curv(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    if '_mask_curvature' in cache:
        curvature = cache['_mask_curvature']
    else:
        curvature = _mask_curvature(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    
    return np.mean(np.abs(curvature))

def nearest_boundary_score(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    if '_mask_curve' in cache:
        curr_boundary = cache['_mask_curve']
    else:
        curr_boundary = _mask_curve(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    curr_boundary = curr_boundary[0].reshape((-1, 2))

    pix_size = (h_pix_size + w_pix_size) / 2
    
    all_dist_quans = []
    for boundary_pt in curr_boundary:
        horizontally_close = np.abs(brain_boundary_pts[:, 0] - boundary_pt[0]) < 250
        vertically_close = np.abs(brain_boundary_pts[:, 1] - boundary_pt[1]) < 250
        close_points = np.logical_and(horizontally_close, vertically_close)
        relevant_brain = brain_boundary_pts[close_points, :]
        distances = np.linalg.norm(relevant_brain - boundary_pt.reshape((1, 2)), axis=1) * pix_size
        if distances.shape[0] == 0:
            distances = np.ones((1000,)) * 500 * pix_size

        nearest_boundaries = np.argsort(distances)
        dist_quantiles = np.quantile(distances[nearest_boundaries[0:1000]], np.arange(0, 1, .1))
        all_dist_quans.append(dist_quantiles)
        
    min_dist_quans = np.amin(np.array(all_dist_quans), axis=0)
    X = np.linspace(0, 1, len(min_dist_quans))
    polyfit_res = np.polyfit(X, min_dist_quans, deg=1, full=True)
    a, _ = polyfit_res[0]
    err = polyfit_res[1][0] / a

    return {'slope': a, 'err': err}

def convex_overlap(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    if '_brain_mask' in cache:
        brain_mask = cache['_brain_mask']
    else:
        brain_mask = _brain_mask(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    
    if '_mask_convex_hull' in cache:
        m_cnx_hull = cache['_mask_convex_hull']
    else:
        m_cnx_hull = _mask_convex_hull(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    
    cnx_filled = np.zeros(brain_mask.shape, np.uint8)
    cv2.fillPoly(cnx_filled, pts=[m_cnx_hull], color=255)

    brain_overlap = cv2.bitwise_and(brain_mask,brain_mask,mask = cnx_filled)

    overlaping_area = mask_area(brain_overlap, h_pix_size, w_pix_size, None, None)
    tumorfree_hull_area = mask_area(cnx_filled, h_pix_size, w_pix_size, None, None) - mask_area(obj_segments, h_pix_size, w_pix_size, None, None)
    overlap_score = overlaping_area / tumorfree_hull_area if tumorfree_hull_area != 0 else 0

    return overlap_score
    

def __r_thetas(thetas, tumor_point, brain_pts):
    # To check the collision, we perform a slope calculation
    deltaYs = brain_pts[:, 0] - tumor_point[0]
    deltaXs = brain_pts[:, 1] - tumor_point[1]
    deltaXs[(0 <= deltaXs) | (deltaXs < EPSILON)] = EPSILON
    deltaXs[(-EPSILON < deltaXs) | (deltaXs <= 0)] = -EPSILON
    m_hat = deltaYs / deltaXs
    m_hats = np.tile(m_hat.reshape((-1, 1)), thetas.shape[0])
    collision_dists = m_hats - np.tan(thetas)

    return np.any(collision_dists < 1e-5, axis=0)

def __v(tumor_point, brain_pts, delta=3e-2):
    thetas = np.arange(0, 2*np.pi, delta)
    return np.sum(__r_thetas(thetas, tumor_point, brain_pts)) * delta / (2*np.pi)

# https://doi.org/10.1016/0031-3203(85)90041-X
def surroundedness_degree(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    if '_brain_mask' in cache:
        brain_mask = cache['_brain_mask']
    else:
        brain_mask = _brain_mask(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    
    if '_mask_curve' in cache:
        mask_curve = cache['_mask_curve']
    else:
        mask_curve = _mask_curve(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    
    pixel_locs = np.where(obj_segments == 255)
    min_y, max_y = np.amin(pixel_locs[0]), np.amax(pixel_locs[0])
    min_x, max_x = np.amin(pixel_locs[1]), np.amax(pixel_locs[1])
    h = max_y - min_y
    w = max_x - min_x

    min_y, min_x = int(max(min_y - h/20, 0)), int(max(min_x - w/20, 0))
    max_y, max_x = int(min(max_y + h/20, brain_mask.shape[0])), int(min(max_x + w/20, brain_mask.shape[1]))

    # print('(x0, y0) -> (x1, y1): (%d, %d) -> (%d, %d)'%(min_x, min_y, max_x, max_y))
    selected_brain = brain_boundary_pts[(min_y < brain_boundary_pts[:, 0]) & (brain_boundary_pts[:, 0] < max_y), :]
    selected_brain = selected_brain[(min_x < selected_brain[:, 1]) & (selected_brain[:, 1] < max_x), :]
    
    tumor_points = mask_curve[0].reshape(-1, 2)
    tumor_points = tumor_points[::4, :]
    v_list = np.array([__v(tumor_point, selected_brain) for tumor_point in tumor_points])

    return np.mean(v_list)
    

def enclosing_circle_overlap(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    if '_brain_mask' in cache:
        brain_mask = cache['_brain_mask']
    else:
        brain_mask = _brain_mask(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)

    if '_mask_curve' in cache:
        mask_curve = cache['_mask_curve']
    else:
        mask_curve = _mask_curve(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    
    ellipse = cv2.fitEllipse(mask_curve[0])
    r = ellipse[2]

    circled_mask = np.zeros(obj_segments.shape, np.uint8)
    circled_mask = cv2.ellipse(circled_mask, (ellipse[0], ellipse[1], r*1.3), 255, int(.3*r))
    
    brain_overlap = cv2.bitwise_and(brain_mask,brain_mask,mask = circled_mask)

    overlaping_area = mask_area(brain_overlap, h_pix_size, w_pix_size, None, None)
    circle_mask_area = mask_area(circled_mask, h_pix_size, w_pix_size, None, None)

    overlap_score = overlaping_area / circle_mask_area

    return overlap_score

def get_interface(brain_mask, tumor_mask, erode_size=15, dilate_size=25, radius=35):
    dilate_kernel = np.ones((dilate_size,dilate_size),np.uint8)
    erode_kernel = np.ones((erode_size,erode_size),np.uint8)
    brain_mask_bound = cv2.erode(brain_mask, erode_kernel, iterations = 1)
    brain_mask_bound = cv2.dilate(brain_mask_bound, dilate_kernel, iterations = radius)
    brain_mask_bound[np.where(brain_mask != 0)] = 0
    tumor_interface = cv2.bitwise_and(tumor_mask, tumor_mask, mask = brain_mask_bound)

    return tumor_interface, brain_mask_bound


src_list = [
    '../data/LB-Dual GFAP CK IHC/LB01.svs',
    '../data/LB-Dual GFAP CK IHC/LB02.svs',
    '../data/LB-Dual GFAP CK IHC/LB03.svs',
    '../data/LB-Dual GFAP CK IHC/LB04.svs',
    '../data/LB-Dual GFAP CK IHC/LB05.svs',
    '../data/LB-Dual GFAP CK IHC/LB06.svs',
    '../data/LB-Dual GFAP CK IHC/LB07.svs',
    '../data/LB-Dual GFAP CK IHC/LB08.svs',
    '../data/LB-Dual GFAP CK IHC/LB09.svs',
    '../data/LB-Dual GFAP CK IHC/LB10.svs',
    '../data/LB-Dual GFAP CK IHC/LB13.svs',
    '../data/LB-Dual GFAP CK IHC/LB14.svs',
    '../data/LB-Dual GFAP CK IHC/LB15.svs',
    '../data/LB-Dual GFAP CK IHC/LB16.svs',
    '../data/LB-Dual GFAP CK IHC/LB17.svs',
    '../data/LB-Dual GFAP CK IHC/LB18.svs',
    '../data/LB-Dual GFAP CK IHC/LB19.svs',
    '../data/LB-Dual GFAP CK IHC/LB20.svs',
    '../data/LB-Dual GFAP CK IHC/LB21.svs',
    '../data/LB-Dual GFAP CK IHC/LB22.svs',
    # '../data/LB-Dual GFAP CK IHC/LB24.svs', ### Necrotic, discard
    '../data/LB-Dual GFAP CK IHC/LB31.svs',
    '../data/LB-Dual GFAP CK IHC/LB34.svs',
    '../data/LB-Dual GFAP CK IHC/LB35.svs',
    '../data/LB-Dual GFAP CK IHC/LB37.svs',
    '../data/LB-Dual GFAP CK IHC/LB40.svs',
    '../data/LB-Dual GFAP CK IHC/LB44.svs',
]

plates_info = None

for src in src_list:
    print('File: %s...'%src)
    pref = src.split('/')[3].split('.')[0]
    for i in range(1):
        print('Loading instance %d...'%(i+1))
        curr_img = read_slide(src, size=(3500, 0), thumbnail_size=(3000, 0))
        print('h_pix_size:', curr_img['h_pix_size'], 'w_pix_size:', curr_img['w_pix_size'])
        
        ########################
        print('Segmenting ...')
        stains, tumor_mask, brain_mask, immune_mask = color_analysis(curr_img)
        
        tumor_interface, brain_mask_bound = get_interface(brain_mask, tumor_mask, erode_size=10, dilate_size=3, radius=25)
        
        seg = np.zeros((*tumor_mask.shape, 3))
        seg[:, :, 0] = tumor_mask
        seg[:, :, 1] = brain_mask
        seg[:, :, 2] = immune_mask
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(curr_img['main'])
        axs[0, 0].set_title('Current Image')
        axs[1, 0].imshow(seg)
        axs[1, 0].set_title('Segmentation')
        seg[:, :, 2] = tumor_interface
        axs[0, 1].imshow(seg)
        axs[0, 1].set_title('Tumor Interface')
        seg[:, :, 2] = brain_mask_bound
        axs[1, 1].imshow(seg)
        axs[1, 1].set_title('Brain Boundary')
        fig.savefig('steps/%s_interface.png'%pref); plt.close() # plt.show()
        # continue
        
        print('Finding connected components ...')
        labeled_img, labels_im, biggest_comps = find_tumors(brain_mask_bound, return_colored_comps=False)

        print('Finding brain boundaries ...')
        brain_boundary_pts, edge_img = find_normal_boundaries(brain_mask)

        print('Extracting features ...')
        all_features = extract_features(
            src,
            stains,
            labels_im,
            biggest_comps,
            [
                _tiled_switch,
                brain_area,
                tumor_area,
                tiled_count,
                tiled_sum,
                # _brain_mask,
                # _mask_curve,
                # _mask_curvature,
                # _mask_convex_hull,
                # mask_perimeter,
                # mask_area,
                # mask_compactness,
                # mask_convexity,
                # mask_solidity,
                # #circular_variance,
                # bending_energy,
                # total_abs_curv,
                # nearest_boundary_score,
                # convex_overlap,
                # surroundedness_degree,
                # enclosing_circle_overlap
            ],
            curr_img['h_pix_size'],
            curr_img['w_pix_size'],
            brain_boundary_pts
        )

        features_summary = all_features.describe(percentiles = np.arange(.2, 1, .2), include = 'all')
        features_summary.index.name = 'summary'
        features_summary.reset_index(inplace=True)
        features_summary = features_summary >> mutate(case = int(src[-6:-4]))
        
        all_features.hist(bins=30, grid=False, xlabelsize=5)
        plt.savefig('steps/%s_features.png'%pref) # plt.show()

        if plates_info is None:
            plates_info = features_summary
        else:
            plates_info = pd.concat([plates_info, features_summary])
            plates_info.to_csv('plates_info_dual.temp.csv', index=False)
        
    print('Done\n')

print(plates_info)
plates_info.to_csv('plates_info_dual.csv', index=False)
