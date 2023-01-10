from email.mime import image
from select import select
from tabnanny import verbose
import slideio
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from random import randint
import pandas as pd
from dfply import *

from sklearn.decomposition import NMF

from tqdm import tqdm


EPSILON = 1e-5

def read_slide(src, rect=None, size=None, thumbnail_size=None):
    brain_slide = slideio.open_slide(src, 'SVS')
    scene = brain_slide.get_scene(0)
    img_size = scene.size
    
    # print('================== Meta Data Info ==================')
    # print('raw_metadata:')
    # print(brain_slide.raw_metadata)

    # print('magnification:')
    # print(scene.magnification)

    # print('resolution:')
    # print(scene.resolution)
    # print('================== ==== ==== ==== ==================')

    if rect is None:
        x0, y0 = 0, 0 # randint(int(img_size[0]*.05), int(img_size[0]*.1)), randint(int(img_size[1]*.05), int(img_size[1]*.1))
        x1, y1 = img_size[0], img_size[1] # randint(int(img_size[0]*.9), int(img_size[0]*.95)), randint(int(img_size[1]*.9), int(img_size[1]*.95))
        rect = (x0, y0, x1, y1)
    if thumbnail_size is None:
        width = min(3000, max(2000, rect[2] - rect[0]))
        thumbnail_size = (width, 0)
    if size is None:
        width = min(9000, max(5000, rect[2] - rect[0]))
        size = (width, 0)
    print("img_size:", img_size)

    print("==============\nreading thumbnail...", thumbnail_size)
    thumbnail = scene.read_block(rect, thumbnail_size)
    # cropped_w = (int(thumbnail.shape[1] * .01), int(thumbnail.shape[1] * .9))
    # cropped_h = (int(thumbnail.shape[0] * .01), int(thumbnail.shape[0] * .9))
    # thumbnail = thumbnail[cropped_h[0]:cropped_h[1], cropped_w[0]:cropped_w[1], 0:3]
    print("thumbnail read", thumbnail.shape, "\n==============")

    print("==============\nreading main image...", size)
    image = scene.read_block(rect, size)
    # cropped_w = (int(image.shape[1] * .01), int(image.shape[1] * .9))
    # cropped_h = (int(image.shape[0] * .01), int(image.shape[0] * .9))
    # image = image[cropped_h[0]:cropped_h[1], cropped_w[0]:cropped_w[1], 0:3]
    print("main image read", image.shape, "\n==============")

    # I am sure about the indices, although they might seem confusing
    h_pix_size = 1e6 * scene.resolution[1] * image.shape[0] / img_size[1]
    w_pix_size = 1e6 * scene.resolution[0] * image.shape[1] / img_size[0]
    
    return {'main': image, 'thumbnail': thumbnail, 'h_pix_size': h_pix_size, 'w_pix_size': w_pix_size}

def color_analysis(image):
    STAIN_CNT = 5

    BG_STAIN = 0
    TUMOR_STAIN = 4
    BRAIN_STAIN = 2
    REST_STAINS = [e for e in list(range(STAIN_CNT)) if e not in [BG_STAIN, TUMOR_STAIN, BRAIN_STAIN]]

    # YCC_img = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
    # LAB_img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    # colors = image.reshape((-1, 3)) / 255.

    main_img, thumbnail = image['main'], image['thumbnail']

    pixels = thumbnail.reshape((-1, 3)) / 255.
    model = NMF(n_components=STAIN_CNT, init='random', random_state=0)
    W = model.fit_transform(pixels)
    H = model.components_
    print(H)

    # weights = W.reshape((thumbnail.shape[0], thumbnail.shape[1], STAIN_CNT))
    # _, axs = plt.subplots(2, 3)
    # axs[0, 0].imshow(thumbnail)
    # axs[0, 0].set_title('thumbnail')
    # for i in range(1, STAIN_CNT+1):
    #     axs[(i // 3), (i % 3)].imshow(np.matmul(weights[:, :, (i-1):i], H[(i-1):i, :]))
    #     axs[(i // 3), (i % 3)].set_title('layer %d'%i)
    # plt.show()

    W[:, REST_STAINS] = 0
    W[:, BRAIN_STAIN] = W[:, BRAIN_STAIN]
    detected_color = np.argmax(W, axis=1)

    # kernel = np.ones((2,2),np.uint8)
    
    tumor_mask = np.zeros(detected_color.shape)
    tumor_mask[detected_color == TUMOR_STAIN] = 1
    tumor_mask = tumor_mask.reshape(thumbnail.shape[0], thumbnail.shape[1])
    # tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, kernel)
    tumor_mask = cv2.bilateralFilter(tumor_mask.astype(np.uint8), 9, 5, 5)

    brain_mask = np.zeros(detected_color.shape)
    brain_mask[detected_color == BRAIN_STAIN] = 1
    brain_mask = brain_mask.reshape(thumbnail.shape[0], thumbnail.shape[1])
    # brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, kernel)
    brain_mask = cv2.bilateralFilter(brain_mask.astype(np.uint8), 9, 5, 5)

    seg1 = np.zeros(thumbnail.shape)
    seg1[:, :, 0] = tumor_mask
    seg1[:, :, 1] = brain_mask

    ##############################################################################################################
    pixels = main_img.reshape((-1, 1, 3)) / 255.
    conv_pix = cv2.cvtColor(np.float32(pixels), cv2.COLOR_RGB2Lab)
    conv_H = cv2.cvtColor(np.float32(H.reshape((-1, 1, 3))), cv2.COLOR_RGB2Lab)
    color_diffs = np.hstack([np.linalg.norm(conv_pix - stain_col, axis=2).reshape((-1,1)) for stain_col in conv_H])
    detected_color2 = np.argmin(color_diffs, axis=1)
    # print("########################################")
    # print(color_diffs.shape, pixels.shape, conv_pix.shape)
    # print(color_diffs)
    # print("########################################")

    kernel2x2 = np.ones((2,2),np.uint8)
    kernel3x3 = np.ones((3,3),np.uint8)
    kernel4x4 = np.ones((4,4),np.uint8)

    tumor_mask2 = np.zeros(detected_color2.shape)
    tumor_mask2[detected_color2 == TUMOR_STAIN] = 1
    tumor_mask2 = tumor_mask2.reshape(main_img.shape[0], main_img.shape[1])
    # tumor_mask2 = cv2.morphologyEx(tumor_mask2, cv2.MORPH_OPEN, kernel)
    tumor_mask2 = cv2.bilateralFilter(tumor_mask2.astype(np.uint8), 9, 5, 5)

    brain_mask2 = np.zeros(detected_color2.shape)
    brain_mask2[detected_color2 == BRAIN_STAIN] = 1
    brain_mask2 = brain_mask2.reshape(main_img.shape[0], main_img.shape[1])
    brain_mask2 = cv2.erode(brain_mask2, kernel2x2, iterations = 1)
    brain_mask2 = cv2.dilate(brain_mask2, kernel3x3, iterations = 2)
    brain_mask2 = cv2.erode(brain_mask2, kernel2x2, iterations = 1)
    brain_mask2 = cv2.dilate(brain_mask2, kernel4x4, iterations = 2)
    # brain_mask2 = cv2.morphologyEx(brain_mask2, cv2.MORPH_OPEN, kernel)
    brain_mask2 = cv2.bilateralFilter(brain_mask2.astype(np.uint8), 9, 5, 5)
    overlaps = cv2.bitwise_and(tumor_mask2, brain_mask2)
    brain_mask2 = brain_mask2 - overlaps

    seg2 = np.zeros(main_img.shape)
    seg2[:, :, 0] = tumor_mask2
    seg2[:, :, 1] = brain_mask2
    ##############################################################################################################
    
    fig, axs = plt.subplots(2, 2)
    fig.tight_layout()
    axs[0, 0].imshow(main_img)
    axs[0, 0].set_title('Main Image')
    axs[0, 1].imshow(seg1)
    axs[0, 1].set_title('Old Segmentation')
    # axs[1, 0].imshow(W[:, [BG_STAIN, BRAIN_STAIN, TUMOR_STAIN]].reshape(thumbnail.shape))
    # axs[1, 0].set_title('Weights')
    axs[1, 1].imshow(seg2)
    axs[1, 1].set_title('New Segmentation')

    # colored_pixs = np.logical_or(detected_color == BRAIN_STAIN, detected_color == TUMOR_STAIN)
    # chosen_index = np.random.randint(colored_pixs.sum(), size=30000)
    # low_cols = (W[colored_pixs,:])[:, [BRAIN_STAIN,TUMOR_STAIN]]
    # weighted_colors = np.hstack([np.zeros((low_cols.shape[0], 1)), low_cols]); weighted_colors[weighted_colors > 1] = 1
    # axs[1, 1].scatter(
    #     low_cols[chosen_index,0],
    #     low_cols[chosen_index,1],
    #     # c=(colors[colored_pixs,:])[chosen_index,:],
    #     c=weighted_colors[chosen_index,:],
    #     alpha=0.75)
    # line = mlines.Line2D([0, 1], [0, 1], color='red')
    # line.set_transform(axs[1, 1].transAxes)
    # axs[1, 1].add_line(line)
    # axs[1, 1].set_title('Color Distribution')

    # tumor_weights = W[detected_color == TUMOR_STAIN, TUMOR_STAIN]
    # brain_weights = W[detected_color == BRAIN_STAIN, BRAIN_STAIN]
    # axs[1, 0].hist(tumor_weights, 300, density=True)
    # axs[1, 0].set_title('Tumor Weights')
    # axs[1, 1].hist(brain_weights, 300, density=True)
    # axs[1, 1].set_title('Brain Weights')
    # axs[1, 1].imshow(W[:, [BG_STAIN, BRAIN_STAIN, TUMOR_STAIN]].reshape(image.shape))
    # axs[1, 1].set_title('Weights_zeroed')
    # axs[1, 0].imshow(tumor_mask, cmap='gray')
    # axs[1, 0].set_title('Tumor')
    # axs[1, 1].imshow(brain_mask, cmap='gray')
    # axs[1, 1].set_title('Brain')
    fig.savefig('steps/%s_seg_res.png'%pref) # plt.show()

    return tumor_mask2, brain_mask2

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

def extract_features(labels_im, biggest_comps, feature_list, h_pix_size, w_pix_size, brain_boundary_pts, max_cmp_cnt=100):
    all_features = []

    # seg = np.zeros((*labels_im.shape, 3))
    # seg[:, :, 0] = tumor_mask
    # seg[:, :, 1] = brain_mask

    
    selection_size = int(max_cmp_cnt / 4)
    rand_g = np.random.default_rng(1)
    main_cmps = np.arange(selection_size)

    sel_ubound = min(200, len(biggest_comps))
    rest_sel_size = min(sel_ubound, max_cmp_cnt) - selection_size
    small_cmps = rand_g.choice(np.arange(selection_size, sel_ubound), rest_sel_size, replace=False)
    selected_indices = np.concatenate((main_cmps, small_cmps))
    selected_indices.sort()
    
    for comp in tqdm(biggest_comps[selected_indices]):
        curr_im = np.zeros(labels_im.shape, np.uint8)
        curr_im[labels_im != comp] = 0
        curr_im[labels_im == comp] = 255

        # seg[labels_im == comp, 2] = 255
        # plt.imshow(seg)
        # plt.show()
        # seg[labels_im == comp, 2] = 0

        if mask_area(curr_im, 1, 1, None, None) < 25:
            break

        curr_features = calculate_features(curr_im, feature_list, h_pix_size, w_pix_size, brain_boundary_pts)
        all_features.append(curr_features)

    all_features = pd.DataFrame.from_dict(all_features)

    return all_features

def calculate_features(object_mask, feature_list, h_pix_size, w_pix_size, brain_boundary_pts, return_hidden=False):
    res = {}
    for feature_func in feature_list:
        feature_val = feature_func(object_mask, h_pix_size, w_pix_size, brain_boundary_pts, res)
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
def _brain_mask(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    return brain_mask

def _mask_curve(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def _mask_curvature(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    if '_mask_curve' in cache:
        contours = cache['_mask_curve']
    else:
        contours = _mask_curve(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache)

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

def _mask_convex_hull(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    if '_mask_curve' in cache:
        contours = cache['_mask_curve']
    else:
        contours = _mask_curve(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    hull = cv2.convexHull(contours[0])

    return hull

def mask_perimeter(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    if '_mask_curve' in cache:
        contours = cache['_mask_curve']
    else:
        contours = _mask_curve(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    
    pix_size = (h_pix_size + w_pix_size) / 2

    return cv2.arcLength(contours[0], True) * pix_size

def mask_area(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    pix_area = h_pix_size * w_pix_size
    return np.where(obj_mask != 0)[0].shape[0] * pix_area

def mask_compactness(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    m_perim = cache['mask_perimeter'] if 'mask_perimeter' in cache else mask_perimeter(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    m_area = cache['mask_area'] if 'mask_area' in cache else mask_area(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    compactness = np.pi * 4 * m_area / m_perim / m_perim

    return compactness

def mask_convexity(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    m_perim = cache['mask_perimeter'] if 'mask_perimeter' in cache else mask_perimeter(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    m_cnx_hull = cache['_mask_convex_hull'] if '_mask_convex_hull' in cache else _mask_convex_hull(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    m_cnx_perim = mask_perimeter(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, {'_mask_curve': [m_cnx_hull]})

    convexity = m_cnx_perim / m_perim

    return convexity

def mask_solidity(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    m_area = cache['mask_area'] if 'mask_area' in cache else mask_area(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    m_cnx_hull = cache['_mask_convex_hull'] if '_mask_convex_hull' in cache else _mask_convex_hull(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    m_cnx_area = cv2.contourArea(m_cnx_hull)

    solidity = m_area / m_cnx_area if m_cnx_area != 0 else 1

    return solidity

def bending_energy(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    if '_mask_curvature' in cache:
        curvature = cache['_mask_curvature']
    else:
        curvature = _mask_curvature(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    
    return np.mean(np.power(curvature, 2))

def total_abs_curv(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    if '_mask_curvature' in cache:
        curvature = cache['_mask_curvature']
    else:
        curvature = _mask_curvature(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    
    return np.mean(np.abs(curvature))

def nearest_boundary_score(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    if '_mask_curve' in cache:
        curr_boundary = cache['_mask_curve']
    else:
        curr_boundary = _mask_curve(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache)
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
        # print('boundary_pt', boundary_pt)
        # print('distances', distances)
        # print('nearest_boundaries', nearest_boundaries)
        # print('dist_quantiles', dist_quantiles)
        # exit()
    min_dist_quans = np.amin(np.array(all_dist_quans), axis=0)
    X = np.linspace(0, 1, len(min_dist_quans))
    polyfit_res = np.polyfit(X, min_dist_quans, deg=1, full=True)
    a, _ = polyfit_res[0]
    err = polyfit_res[1][0] / a
    
    # new_img = np.zeros((*curr_im.shape, 3))
    # new_img[:, :, 0] = curr_im
    # new_img[:, :, 1] = edge_img
    # plt.imshow(new_img)
    # plt.show()

    # print('y = %.2f*x + %.2f, err = %.2f'%(a, b, err/a))
    # plt.scatter(X, min_dist_quans)
    # plt.plot(X, a*X+b, '--')
    # plt.show()

    return {'slope': a, 'err': err}

def convex_overlap(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    if '_brain_mask' in cache:
        brain_mask = cache['_brain_mask']
    else:
        brain_mask = _brain_mask(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    
    if '_mask_convex_hull' in cache:
        m_cnx_hull = cache['_mask_convex_hull']
    else:
        m_cnx_hull = _mask_convex_hull(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    
    cnx_filled = np.zeros(brain_mask.shape, np.uint8)
    cv2.fillPoly(cnx_filled, pts=[m_cnx_hull], color=255)

    brain_overlap = cv2.bitwise_and(brain_mask,brain_mask,mask = cnx_filled)

    # print('brain_overlap(%s): %f'%(brain_overlap.dtype, mask_area(brain_overlap, None, None)))
    # print('cnx_filled(%s): %f'%(cnx_filled.dtype, mask_area(cnx_filled, None, None)))
    # print('obj_mask(%s): %f'%(obj_mask.dtype, mask_area(obj_mask, None, None)))

    overlaping_area = mask_area(brain_overlap, h_pix_size, w_pix_size, None, None)
    tumorfree_hull_area = mask_area(cnx_filled, h_pix_size, w_pix_size, None, None) - mask_area(obj_mask, h_pix_size, w_pix_size, None, None)
    overlap_score = overlaping_area / tumorfree_hull_area if tumorfree_hull_area != 0 else 0

    # visualized = np.zeros((*brain_mask.shape, 3))
    # visualized[:, :, 0] = obj_mask
    # visualized[:, :, 1] = brain_mask
    # visualized[:, :, 2] = cnx_filled * .1
    
    # print(overlap_score)
    # _, axs = plt.subplots(2)
    # axs[0].imshow(obj_mask, cmap='gray')
    # axs[0].set_title('obj_mask')
    # axs[1].imshow(visualized)
    # axs[1].set_title('visualized')
    # plt.show()

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
def surroundedness_degree(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    if '_brain_mask' in cache:
        brain_mask = cache['_brain_mask']
    else:
        brain_mask = _brain_mask(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    
    if '_mask_curve' in cache:
        mask_curve = cache['_mask_curve']
    else:
        mask_curve = _mask_curve(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    
    pixel_locs = np.where(obj_mask == 255)
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

    # visualized = np.zeros((*brain_mask.shape, 3))
    # visualized[:, :, 0] = obj_mask
    # visualized[:, :, 1] = brain_mask
    # visualized[:, :, 2] = cnx_filled * .1
    
    # _, axs = plt.subplots(2)
    # axs[0].imshow(obj_mask, cmap='gray')
    # axs[0].set_title('obj_mask')
    # axs[1].imshow(visualized)
    # axs[1].set_title('visualized')
    # plt.show()

    return np.mean(v_list)
    

def enclosing_circle_overlap(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    if '_brain_mask' in cache:
        brain_mask = cache['_brain_mask']
    else:
        brain_mask = _brain_mask(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache)

    if '_mask_curve' in cache:
        mask_curve = cache['_mask_curve']
    else:
        mask_curve = _mask_curve(obj_mask, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    
    # (x,y), r = cv2.minEnclosingCircle(mask_curve[0])
    ellipse = cv2.fitEllipse(mask_curve[0])
    r = ellipse[2]

    circled_mask = np.zeros(obj_mask.shape, np.uint8)
    # circled_mask = cv2.circle(circled_mask, (int(x), int(y)), int(r*1.3), 255, int(.3*r))
    circled_mask = cv2.ellipse(circled_mask, (ellipse[0], ellipse[1], r*1.3), 255, int(.3*r))
    
    brain_overlap = cv2.bitwise_and(brain_mask,brain_mask,mask = circled_mask)

    overlaping_area = mask_area(brain_overlap, h_pix_size, w_pix_size, None, None)
    circle_mask_area = mask_area(circled_mask, h_pix_size, w_pix_size, None, None)

    overlap_score = overlaping_area / circle_mask_area

    # visualized = np.zeros((*brain_mask.shape, 3))
    # visualized[:, :, 0] = obj_mask
    # visualized[:, :, 1] = brain_mask
    # visualized[:, :, 2] = circled_mask
    
    # print(overlap_score)
    # _, axs = plt.subplots(2)
    # axs[0].imshow(obj_mask, cmap='gray')
    # axs[0].set_title('obj_mask')
    # axs[1].imshow(visualized)
    # axs[1].set_title('visualized')
    # plt.show()

    return overlap_score


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
        curr_img = read_slide(src, size=(10000, 0), thumbnail_size=(3500, 0))
        print('h_pix_size:', curr_img['h_pix_size'], 'w_pix_size:', curr_img['w_pix_size'])
        
        ########################
        print('Segmenting ...')
        # segmented_image = segment(curr_img, .1, 100, 200)
        # segmented_image = segmented_image.astype(np.uint8)
        tumor_mask, brain_mask = color_analysis(curr_img)
        
        kernel15x15 = np.ones((15,15),np.uint8)
        kernel25x25 = np.ones((25,25),np.uint8)
        brain_mask_bound = cv2.erode(brain_mask, kernel25x25, iterations = 1)
        brain_mask_bound = cv2.dilate(brain_mask_bound, kernel15x15, iterations = 35)
        brain_mask_bound[np.where(brain_mask != 0)] = 0
        tumor_interface = cv2.bitwise_and(tumor_mask, tumor_mask, mask = brain_mask_bound)
        
        seg = np.zeros((*tumor_mask.shape, 3))
        seg[:, :, 0] = tumor_mask
        seg[:, :, 1] = brain_mask
        seg[:, :, 2] = brain_mask_bound
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(curr_img['main'])
        axs[0, 0].set_title('curr_img')
        axs[1, 0].imshow(seg)
        axs[1, 0].set_title('brain_mask_bound')
        seg[:, :, 2] = tumor_interface
        axs[0, 1].imshow(seg)
        axs[0, 1].set_title('seg')
        fig.savefig('steps/%s_interface.png'%pref) # plt.show()
        # continue
        
        print('Finding connected components ...')
        labeled_img, labels_im, biggest_comps = find_tumors(tumor_interface, return_colored_comps=False)

        print('Finding brain boundaries ...')
        brain_boundary_pts, edge_img = find_normal_boundaries(brain_mask)

        print('Extracting features ...')
        all_features = extract_features(
            labels_im,
            biggest_comps,
            [
                _brain_mask,
                _mask_curve,
                _mask_curvature,
                _mask_convex_hull,
                mask_perimeter,
                mask_area,
                mask_compactness,
                mask_convexity,
                mask_solidity,
                #circular_variance,
                bending_energy,
                total_abs_curv,
                nearest_boundary_score,
                convex_overlap,
                surroundedness_degree,
                enclosing_circle_overlap
            ],
            curr_img['h_pix_size'],
            curr_img['w_pix_size'],
            brain_boundary_pts
        )

        # print(all_features)

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
