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
            w, h = max(10, min(img_size[0] - x0 - 10, w)), max(10, min(img_size[1] - y0 - 10, h))
            rect = (x0, y0, w, h)

    if thumbnail_size is not None:
        thumbnail = scene.read_block(rect, thumbnail_size)
    else:
        thumbnail = None
    if size is None:
        width = min(9000, max(5000, rect[2]))
        size = (width, 0)
    # print("img_size:", img_size)

    # print("==============\nreading main image...", size)
    image = scene.read_block(rect, size)
    # print("main image read", image.shape, "\n==============")

    # I am sure about the indices, although they might seem confusing
    h_pix_size = 1e6 * scene.resolution[1] * rect[3] / image.shape[0]
    w_pix_size = 1e6 * scene.resolution[0] * rect[2] / image.shape[1]
    
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

def colorbased_seg(main_img, stains, strict = False, kern_s = 2, kern_m = 3, kern_b = 4):
    pixels = main_img.reshape((-1, 1, 3)) / 255.
    conv_pix = cv2.cvtColor(np.float32(pixels), cv2.COLOR_RGB2Lab)
    conv_stains = cv2.cvtColor(np.float32(stains.reshape((-1, 1, 3))), cv2.COLOR_RGB2Lab)
    color_diffs = np.hstack([np.linalg.norm(conv_pix - stain_col, axis=2).reshape((-1,1)) for stain_col in conv_stains])
    detected_color = np.argmin(color_diffs, axis=1)

    if strict:
        middle_color = (color_diffs[:, TUMOR_STAIN] / color_diffs[:, BRAIN_STAIN] < 2) & (detected_color == BRAIN_STAIN) & (color_diffs[:, TUMOR_STAIN] < 40)
        detected_color[middle_color] = BG_STAIN

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

    # colordiff2d = color_diffs.reshape((*main_img.shape[0:2], stains.shape[0])).astype(np.uint8)

    # _, axs = plt.subplots(3, 2)
    # axs[0, 0].imshow(seg)
    # axs[0, 0].set_title('Current Segmentation')
    # axs[0, 1].hist(colordiff2d[:,:,0].reshape((-1,)), bins=50)
    # axs[0, 1].set_title('Stain 0')
    # axs[1, 0].hist(colordiff2d[:,:,1].reshape((-1,)), bins=50)
    # axs[1, 0].set_title('Stain 1')
    # axs[1, 1].hist(colordiff2d[:,:,2].reshape((-1,)), bins=50)
    # axs[1, 1].set_title('Stain 2')
    # axs[2, 0].hist(colordiff2d[:,:,3].reshape((-1,)), bins=50)
    # axs[2, 0].set_title('Stain 3')
    # axs[2, 1].hist(colordiff2d[:,:,4].reshape((-1,)), bins=50)
    # axs[2, 1].set_title('Stain 4')

    # def update_hist(_):
    #     x0 = axs[0, 0].get_xlim()[0]
    #     y0 = axs[0, 0].get_ylim()[1]
    #     x1 = axs[0, 0].get_xlim()[1]
    #     y1 = axs[0, 0].get_ylim()[0]
    #     x0,y0, x1,y1 = np.round([x0, y0, x1, y1]).astype(int)
    #     zoomed_diffs = colordiff2d[y0:y1, x0:x1, :]
    #     axs[0, 1].cla()
    #     axs[0, 1].hist(zoomed_diffs[:,:,0].reshape((-1,)), bins=50)
    #     axs[0, 1].set_title('Stain 0')
    #     axs[0, 1].figure.canvas.draw_idle()
    #     axs[1, 0].cla()
    #     axs[1, 0].hist(zoomed_diffs[:,:,1].reshape((-1,)), bins=50)
    #     axs[1, 0].set_title('Stain 1')
    #     axs[1, 0].figure.canvas.draw_idle()
    #     axs[1, 1].cla()
    #     axs[1, 1].hist(zoomed_diffs[:,:,2].reshape((-1,)), bins=50)
    #     axs[1, 1].set_title('Stain 2')
    #     axs[1, 1].figure.canvas.draw_idle()
    #     axs[2, 0].cla()
    #     axs[2, 0].hist(zoomed_diffs[:,:,3].reshape((-1,)), bins=50)
    #     axs[2, 0].set_title('Stain 3')
    #     axs[2, 0].figure.canvas.draw_idle()
    #     axs[2, 1].cla()
    #     axs[2, 1].hist(zoomed_diffs[:,:,4].reshape((-1,)), bins=50)
    #     axs[2, 1].set_title('Stain 4')
    #     axs[2, 1].figure.canvas.draw_idle()
    # axs[0, 0].callbacks.connect('xlim_changed', update_hist)
    # axs[0, 0].callbacks.connect('ylim_changed', update_hist)

    # plt.show()

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

def extract_features(src, stains, labels_im, biggest_comps, feature_list, brain_boundary_pts, max_cmp_cnt=20):
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

    total_tiles = 0
    
    for comp in tqdm(biggest_comps[selected_indices]):
        # skipping small components
        comp_size = np.where(labels_im == comp)[0].shape[0]
        if comp_size < 200:
                break

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

        # Partitioning big brain components
        dh = h / np.ceil(h / .1)
        dw = w / np.ceil(w / .1)
        breaks = np.mgrid[0:w:dw, 0:h:dh].reshape(2,-1).T
        start_pos = [min_x, min_y] + breaks
        windows = np.repeat([[dw, dh]], start_pos.shape[0], axis=0)
        all_rects = np.hstack([start_pos, windows])

        for curr_rect in all_rects:
            curr_tumor = read_slide(src, rect = curr_rect, size=(4000, 0))
            h_pix_size, w_pix_size = curr_tumor['h_pix_size'], curr_tumor['w_pix_size']

            # Segmenting high power tumor
            hp_seg, _, _, _ = colorbased_seg(curr_tumor['main'], stains, strict=True, kern_s=4, kern_m=6, kern_b=8)
            hp_seg = denoise_segments(hp_seg, kernel = np.ones((10, 10),np.uint8))

            # skipping tiles with no tumor / brain
            tumor_size = np.where(hp_seg[:,:,0] != 0)[0].shape[0]
            brain_size = np.where(hp_seg[:,:,1] != 0)[0].shape[0]
            if (tumor_size < 22500) or (brain_size < 250000):
                    continue
            # print('total_tiles:', total_tiles, 'tumor_size:', tumor_size, 'brain_size:', brain_size)

            # seg[labels_im == comp, 2] = .4
            
            # fig, axs = plt.subplots(2, 2)
            # fig.tight_layout()
            # axs[0, 0].imshow(curr_img['main'])
            # axs[0, 0].set_title('Current Image')
            # int_rect = np.array([curr_rect[0] * labels_im.shape[1], curr_rect[1] * labels_im.shape[0], curr_rect[2] * labels_im.shape[1], curr_rect[3] * labels_im.shape[0]], np.uint32)
            # empty_canvas = np.zeros(seg.shape[0:2],  np.float64)
            # seg[:, :, 2] += cv2.rectangle(empty_canvas, int_rect[0:2], int_rect[0:2] + int_rect[2:4], .6, 25)
            # axs[0, 1].imshow(seg)
            # axs[0, 1].set_title('Segmented')
            # axs[1, 0].imshow(curr_tumor['main'])
            # axs[1, 0].set_title('Current Tumor Component')
            # axs[1, 1].imshow(hp_seg)
            # axs[1, 1].set_title('High Power Segmented')
            # fig.savefig('debug/%s_%d.png'%(pref, total_tiles)); plt.close() #plt.show()
            # seg[:, :, 2] = 0

            curr_features = calculate_features(hp_seg, feature_list, h_pix_size, w_pix_size, brain_boundary_pts)
            all_features.append(curr_features)
            total_tiles += 1

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

def denoise_segments(segments, kernel):
    segments[:,:,0] = cv2.morphologyEx(segments[:,:,0], cv2.MORPH_OPEN, kernel)
    segments[:,:,0] = cv2.morphologyEx(segments[:,:,0], cv2.MORPH_CLOSE, kernel)
    segments[:,:,1] = cv2.morphologyEx(segments[:,:,1], cv2.MORPH_OPEN, kernel)
    segments[:,:,1] = cv2.morphologyEx(segments[:,:,1], cv2.MORPH_CLOSE, kernel)

    return segments

def make_mask_compact(obj_mask, thickness=200, comprehensive = 20):
    close_kernel = np.ones((thickness,thickness),np.uint8)
    
    _, labels_im, stats, _ = cv2.connectedComponentsWithStats(obj_mask.astype(np.uint8))
    all_areas = stats[:, 4]

    biggest_comps = (-all_areas).argsort()[1:comprehensive]
    all_comps = np.unique(labels_im)
    final_result = obj_mask.copy()
    for comp in all_comps:
        if comp in biggest_comps:
            curr_component = np.zeros(final_result.shape, np.uint8)
            curr_component[labels_im == comp] = 255
            curr_component = cv2.morphologyEx(curr_component, cv2.MORPH_CLOSE, close_kernel).astype(np.uint8)
            final_result[curr_component != 0] = curr_component[curr_component != 0]
    
    return final_result

######## Features
def invasion_count(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
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


    _, bin_t_switch = cv2.threshold(tumor_switch, .3, 255, cv2.THRESH_BINARY)
    _, bin_b_switch = cv2.threshold(brain_switch, .3, 255, cv2.THRESH_BINARY)

    dilate_kernel = np.ones((5,5),np.uint8)
    bin_t_switch = cv2.dilate(bin_t_switch, dilate_kernel, iterations = 1).astype(np.uint8)
    bin_b_switch = cv2.dilate(bin_b_switch, dilate_kernel, iterations = 1).astype(np.uint8)

    switch_thresh = cv2.bitwise_and(bin_b_switch, bin_b_switch, mask = bin_t_switch)
    pooled_segments = np.zeros((*switch_thresh.shape, 3), np.uint8)
    pooled_segments[:,:,0] = bin_tile_tumor * 255
    pooled_segments[:,:,1] = bin_tile_brain * 255
    # switch_image = cv2.bitwise_and(pooled_segments, pooled_segments, mask = switch_thresh)

    contours, hierarchy = cv2.findContours(switch_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return {'cnt': 0, 'min_area': 0, 'max_area': 0, 'med_area': 0}

    child_contour = hierarchy[0, :,2]
    contours_length = np.array(list(map(lambda x: cv2.arcLength(x, True), contours)))
    contours_length[contours_length < 3] = 3
    contours_area = np.array(list(map(cv2.contourArea, contours)))
    contours_compactness = (4 * np.pi * contours_area) / (contours_length ** 2)
    contours = [contours[i] for i in range(len(contours)) if contours_compactness[i] > .5 and child_contour[i] == -1 and contours_area[i] > 50]

    tumor_contours = []
    
    for contour in contours:
        invasion_mask = np.zeros(switch_thresh.shape, np.uint8)
        cv2.drawContours(invasion_mask, [contour], -1, 255, -1)
        invasion_mask = cv2.erode(invasion_mask, dilate_kernel, iterations = 1).astype(np.uint8)
        detected_invasion = cv2.bitwise_and(pooled_segments, pooled_segments, mask = invasion_mask)
        tumor_area = mask_area_px_sq(detected_invasion[:,:,0])
        brain_area = mask_area_px_sq(detected_invasion[:,:,1])
        if tumor_area > brain_area:
            tumor_contours.append(contour)
    
    if len(tumor_contours) == 0:
        return {'cnt': 0, 'min_area': 0, 'max_area': 0, 'med_area': 0}
    
    pix_area = h_pix_size * w_pix_size
    contour_areas = np.array(list(map(cv2.contourArea, tumor_contours))) * pix_area
    
    # found_switches = pooled_segments.copy()
    # cv2.drawContours(found_switches, tumor_contours, -1, (0, 0, 255), 5)
    
    # fig, axs = plt.subplots(2, 2)
    # fig.tight_layout()
    # axs[0, 0].imshow(obj_segments)
    # axs[0, 0].set_title('Segments')
    # axs[0, 1].imshow(switch_image)
    # axs[0, 1].set_title('Switch Image')
    # axs[1, 0].imshow(switch_thresh, cmap='gray')
    # axs[1, 0].set_title('Switch Thresholds')
    # axs[1, 1].imshow(found_switches)
    # axs[1, 1].set_title('Found Switches')
    # plt.show()
    
    return {'cnt': len(tumor_contours), 'min_area': np.min(contour_areas), 'max_area': np.max(contour_areas), 'med_area': np.median(contour_areas)}

def delaunay_score(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    pix_dim = (h_pix_size + w_pix_size) / 2
    fill_number = pix_dim * (obj_segments.shape[0] + obj_segments.shape[1])/2

    open_kernel = np.ones((20,20),np.uint8)
    denoised_brain = cv2.morphologyEx(obj_segments[:,:,1] * 255, cv2.MORPH_OPEN, open_kernel).astype(np.uint8)
    brain_holes = make_mask_compact(denoised_brain)
    brain_holes[denoised_brain != 0] = 0

    open_kernel = np.ones((15,15),np.uint8)
    close_kernel = np.ones((30,30),np.uint8)
    denoised_tumor = cv2.morphologyEx(obj_segments[:,:,0] * 255, cv2.MORPH_OPEN, open_kernel).astype(np.uint8)
    compact_tumor = cv2.morphologyEx(denoised_tumor, cv2.MORPH_CLOSE, close_kernel).astype(np.uint8)

    invaded_tumors = cv2.bitwise_and(compact_tumor, compact_tumor,mask = brain_holes)

    contours, hierarchy = cv2.findContours(invaded_tumors, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return {'min': fill_number, 'quartile1': fill_number, 'median': fill_number, 'quartile3': fill_number, 'max': fill_number}

    child_contour = hierarchy[0, :,2]
    contours_area = np.array(list(map(cv2.contourArea, contours)))
    contours = [contours[i] for i in range(len(contours)) if child_contour[i] == -1 and contours_area[i] > 50]

    if len(contours) < 3:
        return {'min': fill_number, 'quartile1': fill_number, 'median': fill_number, 'quartile3': fill_number, 'max': fill_number}
    
    points = []
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            points.append((cx, cy))
    points = np.array(points, np.float32)

    if len(contours) == 3:
        all_dists = np.linalg.norm(points[[0,1,2],:] - points[[1,2,0],:], axis=1)
        quantiles = np.quantile(all_dists, [0, .25, .5, .75, 1])
        return {'min': quantiles[0], 'quartile1': quantiles[1], 'median': quantiles[2], 'quartile3': quantiles[3], 'max': quantiles[4]}

    rect = cv2.boundingRect(points)
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)
    triangleList = subdiv.getTriangleList()

    all_dists = []
    for t in triangleList:
        head_points = np.array([
            [t[0], t[1]],
            [t[2], t[3]],
            [t[4], t[5]]
        ])
        tail_points = np.array([
            [t[2], t[3]],
            [t[4], t[5]],
            [t[0], t[1]]
        ])
        dists = np.linalg.norm(head_points - tail_points, axis=1)

        all_dists.append(dists)
    
    if len(all_dists) == 0:
        all_dists = np.linalg.norm(points[[0,1,2],:] - points[[1,2,0],:], axis=1)
        quantiles = np.quantile(all_dists, [0, .25, .5, .75, 1])
        return {'min': quantiles[0], 'quartile1': quantiles[1], 'median': quantiles[2], 'quartile3': quantiles[3], 'max': quantiles[4]}

    all_dists = np.array(all_dists).reshape((-1,))

    all_dists = all_dists * pix_dim
    
    quantiles = np.quantile(all_dists, [0, .25, .5, .75, 1])

    # plt.imshow(obj_segments)
    # for t in triangleList:
    #     pt1 = (t[0], t[1])
    #     pt2 = (t[2], t[3])
    #     pt3 = (t[4], t[5])
    #     plt.plot([pt1[0], pt2[0], pt3[0], pt1[0]], [pt1[1], pt2[1], pt3[1], pt1[1]])

    # plt.show()
    
    return {'min': quantiles[0], 'quartile1': quantiles[1], 'median': quantiles[2], 'quartile3': quantiles[3], 'max': quantiles[4]}

def tumor_components(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    open_kernel = np.ones((20,20),np.uint8)
    denoised_brain = cv2.morphologyEx(obj_segments[:,:,1] * 255, cv2.MORPH_OPEN, open_kernel).astype(np.uint8)
    brain_holes = make_mask_compact(denoised_brain)
    brain_holes[denoised_brain != 0] = 0

    open_kernel = np.ones((15,15),np.uint8)
    close_kernel = np.ones((30,30),np.uint8)
    denoised_tumor = cv2.morphologyEx(obj_segments[:,:,0] * 255, cv2.MORPH_OPEN, open_kernel).astype(np.uint8)
    compact_tumor = cv2.morphologyEx(denoised_tumor, cv2.MORPH_CLOSE, close_kernel).astype(np.uint8)

    invaded_tumors = cv2.bitwise_and(compact_tumor, compact_tumor,mask = brain_holes)

    contours, hierarchy = cv2.findContours(invaded_tumors, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return {'cnt': 0, 'min_area': 0, 'max_area': 0, 'med_area': 0}

    child_contour = hierarchy[0, :,2]
    contours_area = np.array(list(map(cv2.contourArea, contours)))
    contours = [contours[i] for i in range(len(contours)) if child_contour[i] == -1 and contours_area[i] > 50]

    if len(contours) == 0:
        return {'cnt': 0, 'min_area': 0, 'max_area': 0, 'med_area': 0}
    
    pix_area = h_pix_size * w_pix_size
    contour_areas = np.array(list(map(cv2.contourArea, contours))) * pix_area
    
    # found_components = obj_segments.copy()
    # cv2.drawContours(found_components, contours, -1, (0, 0, 255), 50)
    
    # fig, axs = plt.subplots(2, 2)
    # fig.tight_layout()
    # axs[0, 0].imshow(obj_segments)
    # axs[0, 0].set_title('Segments')
    # # axs[0, 1].imshow(switch_image)
    # # axs[0, 1].set_title('Switch Image')
    # # axs[1, 0].imshow(switch_thresh, cmap='gray')
    # # axs[1, 0].set_title('Switch Thresholds')
    # axs[1, 1].imshow(found_components)
    # axs[1, 1].set_title('Found components')
    # plt.show()
    
    return {'cnt': len(contours), 'min_area': np.min(contour_areas), 'max_area': np.max(contour_areas), 'med_area': np.median(contour_areas)}

def interface_ellipse(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    # TODO: IMPOOOOOORTAAAANT should be renamed and refactored
    tumor_mask = (obj_segments[:,:,0] * 255).astype(np.uint8)
    brain_mask = (obj_segments[:,:,1] * 255).astype(np.uint8)
    open_kernel = np.ones((20,20),np.uint8)
    dilate_kernel = np.ones((50,50),np.uint8)
    denoised_tumor = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, open_kernel).astype(np.uint8)
    extended_tumor = cv2.dilate(denoised_tumor, dilate_kernel, iterations=1).astype(np.uint8)
    denoised_brain = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, open_kernel).astype(np.uint8)
    extended_brain = cv2.dilate(denoised_brain, dilate_kernel, iterations=1).astype(np.uint8)

    detected_interface = cv2.bitwise_and(extended_tumor, extended_tumor, mask = extended_brain)

    contours, _  = cv2.findContours(detected_interface, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # final_contours = []
    final_eccentricity = []
    final_perimeters = []

    pix_size = (h_pix_size + w_pix_size) / 2

    for cnt in contours:
        if(cnt.shape[0] < 5): continue
        contour_area = cv2.contourArea(cnt)
        if(contour_area < 20): continue
        contour_perim = cv2.arcLength(cnt, True)
        stretch_level = (contour_perim**2) / (4 * np.pi * contour_area)
        final_eccentricity.append(stretch_level)
        final_perimeters.append(contour_perim * pix_size)

    quantiles = [.1, .5, .9]
    eccent_quants = np.quantile(final_eccentricity, quantiles)
    area_quants = np.quantile(final_perimeters, quantiles)

    return {
        **{'eccent_quants_' + str(quantiles[i]): eccent_quants[i] for i in range(len(quantiles))},
        **{'area_quants_' + str(quantiles[i]): area_quants[i] for i in range(len(quantiles))},
    }

def cooccurrence_mat(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    pool_d = 50

    cont_tile_tumor = skimage.measure.block_reduce(obj_segments[:, :, 0], pool_d, np.mean)
    cont_tile_brain = skimage.measure.block_reduce(obj_segments[:, :, 1], pool_d, np.mean)

    unwanted_tiles = np.logical_and(cont_tile_tumor < .1, cont_tile_brain < .1)
    cont_tile_tumor[unwanted_tiles] = 1e-3
    cont_tile_brain[unwanted_tiles] = 1e-3

    proportion = cont_tile_brain / (cont_tile_brain + cont_tile_tumor)

    unwanted_up = unwanted_tiles[:-1, :]
    base4up = (proportion[:-1, :])[~unwanted_up]
    proportion_up = (proportion[1:, :])[~unwanted_up]
    pair_up = np.concatenate((base4up.reshape((-1, 1)), proportion_up.reshape((-1, 1))), axis=1)

    unwanted_left = unwanted_tiles[:, :-1]
    base4left = (proportion[:, :-1])[~unwanted_left]
    proportion_left = (proportion[:, 1:])[[~unwanted_left]]
    pair_left = np.concatenate((base4left.reshape((-1, 1)), proportion_left.reshape((-1, 1))), axis=1)

    all_pairs = np.vstack((pair_up, pair_left))
    all_pairs = np.digitize(all_pairs, [-.1, .33, .67, 1.1]) - 1
    all_unique_pairs, all_unique_counts = np.unique(all_pairs, axis=0, return_counts=True)
    all_unique_counts = all_unique_counts

    result = np.zeros((3, 3))
    result[all_unique_pairs[:, 0], all_unique_pairs[:, 1]] = all_unique_counts
    result = result.reshape((-1,))[1:-1]
    result = result / np.sum(result)
    
    # fig, axs = plt.subplots(2)
    # fig.tight_layout()
    # axs[0].imshow(obj_segments)
    # axs[0].set_title('Segments')
    # axs[1].imshow(result.reshape((1, -1)), cmap='gray')
    # axs[1].set_title('Co-occurrence')
    # plt.show()
    
    return [*result]


def mask_area_px_sq(mask):
    return np.where(mask != 0)[0].shape[0]

def whole_brain_area(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    pix_area = h_pix_size * w_pix_size
    return mask_area_px_sq(obj_segments[:, :, 1]) * pix_area

def whole_tumor_area(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    pix_area = h_pix_size * w_pix_size
    return mask_area_px_sq(obj_segments[:, :, 0]) * pix_area

def _brain_largest_area(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    if '_brain_curve' in cache:
        contours = cache['_brain_curve']
    else:
        contours = _brain_curve(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    
    biggest_comp_mask = np.zeros(obj_segments.shape[:2], np.uint8)
    cv2.drawContours(biggest_comp_mask, (contours,), -1, 1, -1)
    biggest_comp = cv2.bitwise_and(obj_segments, obj_segments,mask = biggest_comp_mask)

    pix_area = h_pix_size * w_pix_size
    return mask_area_px_sq(biggest_comp) * pix_area

def tiled_count(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    pix_area = h_pix_size * w_pix_size
    
    if 'whole_brain_area' in cache:
        brain_area = cache['whole_brain_area'] / pix_area
    else:
        brain_area = whole_brain_area(obj_segments, 1, 1, brain_boundary_pts, cache)
    
    if 'whole_tumor_area' in cache:
        tumor_area = cache['whole_tumor_area'] / pix_area
    else:
        tumor_area = whole_tumor_area(obj_segments, 1, 1, brain_boundary_pts, cache)
    
    if 'invasion_count' in cache:
        tiled_switch = cache['invasion_count']
    else:
        tiled_switch = invasion_count(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)

    return np.where(tiled_switch != 0)[0].shape[0] / (np.sqrt(brain_area) + np.sqrt(tumor_area))

def tiled_sum(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    pix_area = h_pix_size * w_pix_size
    
    if 'whole_brain_area' in cache:
        brain_area = cache['whole_brain_area'] / pix_area
    else:
        brain_area = whole_brain_area(obj_segments, 1, 1, brain_boundary_pts, cache)
    
    if 'whole_tumor_area' in cache:
        tumor_area = cache['whole_tumor_area'] / pix_area
    else:
        tumor_area = whole_tumor_area(obj_segments, 1, 1, brain_boundary_pts, cache)
    
    if 'invasion_count' in cache:
        tiled_switch = cache['invasion_count']
    else:
        tiled_switch = invasion_count(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)

    return np.sum(tiled_switch) / (np.sqrt(brain_area) + np.sqrt(tumor_area))

def _brain_curve(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    contours, _ = cv2.findContours(obj_segments[:,:,1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_cont = max(contours, key = cv2.contourArea)

    return largest_cont

def _mask_curvature(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    if '_brain_curve' in cache:
        contours = cache['_brain_curve']
    else:
        contours = _brain_curve(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)

    conversion_mat = np.array([
        [5/4, 1, 3/4],
        [3/2, 0, 1/2],
        [7/4, 0, 1/4]
    ])

    variation = contours[1:, 0] - contours[0][:-1, 0]
    variation_ang = conversion_mat[variation[:,0]+1, variation[:,1]+1]
    curvature = variation_ang[1:] - variation_ang[:-1]
    curvature[curvature >= 2] = curvature[curvature >= 2] - 2
    curvature[curvature <= -2] = curvature[curvature <= -2] + 2
    curvature = curvature * np.pi

    return curvature

def _brain_convex_hull(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    if '_brain_curve' in cache:
        contours = cache['_brain_curve']
    else:
        contours = _brain_curve(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    hull = cv2.convexHull(contours)

    return hull

def _brain_perimeter(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    if '_brain_curve' in cache:
        contours = cache['_brain_curve']
    else:
        contours = _brain_curve(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    
    pix_size = (h_pix_size + w_pix_size) / 2

    return cv2.arcLength(contours, True) * pix_size

def brain_compactness(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    b_perim = cache['_brain_perimeter'] if '_brain_perimeter' in cache else _brain_perimeter(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    b_area = cache['_brain_largest_area'] if '_brain_largest_area' in cache else _brain_largest_area(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    compactness = np.pi * 4 * b_area / b_perim / b_perim

    return compactness

def brain_convexity(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    b_perim = cache['_brain_perimeter'] if '_brain_perimeter' in cache else _brain_perimeter(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    b_cnx_hull = cache['_brain_convex_hull'] if '_brain_convex_hull' in cache else _brain_convex_hull(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)

    b_cnx_perim = _brain_perimeter(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, {'_brain_curve': b_cnx_hull})

    convexity = b_cnx_perim / b_perim

    return convexity

def brain_solidity(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    b_area = cache['_brain_largest_area'] if '_brain_largest_area' in cache else _brain_largest_area(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    b_cnx_hull = cache['_brain_convex_hull'] if '_brain_convex_hull' in cache else _brain_convex_hull(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    
    pix_area = h_pix_size * w_pix_size
    b_cnx_area = cv2.contourArea(b_cnx_hull) * pix_area

    solidity = b_area / b_cnx_area if b_cnx_area != 0 else 1

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
    if '_brain_curve' in cache:
        curr_boundary = cache['_brain_curve']
    else:
        curr_boundary = _brain_curve(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
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
    b_area = cache['_brain_largest_area'] if '_brain_largest_area' in cache else _brain_largest_area(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    pix_area = h_pix_size * w_pix_size
    b_area = int(b_area / pix_area)

    if '_brain_convex_hull' in cache:
        b_cnx_hull = cache['_brain_convex_hull']
    else:
        b_cnx_hull = _brain_convex_hull(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    
    tumor_mask = obj_segments[:,:,0]
    brain_mask = obj_segments[:,:,1]
    
    cnx_filled = np.zeros(tumor_mask.shape, np.uint8)
    cv2.fillPoly(cnx_filled, pts=[b_cnx_hull], color=255)

    brain_overlap = cv2.bitwise_and(tumor_mask,tumor_mask,mask = cnx_filled)

    overlaping_area = mask_area_px_sq(brain_overlap)
    nobrain_area = mask_area_px_sq(cnx_filled) - b_area
    overlap_score = overlaping_area / nobrain_area if nobrain_area != 0 else 0

    return overlap_score
    
def filled_overlap(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache):
    tumor_mask = obj_segments[:,:,0] * 255
    brain_mask = obj_segments[:,:,1] * 255

    kernel = np.ones((10, 10),np.uint8)
    denoised_brain = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, kernel).astype(np.uint8)
    denoised_tumor = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, kernel).astype(np.uint8)

    kernel = np.ones((40, 40),np.uint8)
    brain_closing = cv2.morphologyEx(denoised_brain, cv2.MORPH_CLOSE, kernel)
    # brain_closing = brain_closing.astype(np.uint8)
    brain_closing[brain_mask != 0] = 0

    brain_overlap = cv2.bitwise_and(denoised_tumor,denoised_tumor,mask = brain_closing)

    contours, hierarchy = cv2.findContours(brain_overlap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return {'cnt': 0, 'min_area': 0, 'max_area': 0, 'med_area': 0}

    parent_contour = hierarchy[0, :,3]
    contours_area = np.array(list(map(cv2.contourArea, contours)))
    area_threshold = 625
    contours = [contours[i] for i in range(len(contours)) if parent_contour[i] == -1 and contours_area[i] > area_threshold]

    if len(contours) == 0:
        return {'cnt': 0, 'min_area': 0, 'max_area': 0, 'med_area': 0}
    
    pix_area = h_pix_size * w_pix_size
    contour_areas = np.array(list(map(cv2.contourArea, contours))) * pix_area
    
    # found_filled_gaps = obj_segments.copy()
    # cv2.drawContours(found_filled_gaps, contours, -1, (0, 0, 255), -1)

    # closed_view = obj_segments.copy()
    # closed_view[:,:,2] = brain_overlap

    # fig, axs = plt.subplots(2, 2)
    # fig.tight_layout()
    # axs[0, 0].imshow(obj_segments)
    # axs[0, 0].set_title('Segments')
    # axs[0, 1].imshow(closed_view)
    # axs[0, 1].set_title('Closed Brain')
    # # axs[1, 0].imshow(switch_thresh, cmap='gray')
    # # axs[1, 0].set_title('Switch Thresholds')
    # axs[1, 1].imshow(found_filled_gaps)
    # axs[1, 1].set_title('Found Filled Gaps')
    # plt.show()

    return {'cnt': len(contours), 'min_area': np.min(contour_areas), 'max_area': np.max(contour_areas), 'med_area': np.median(contour_areas)}

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
    
    if '_brain_curve' in cache:
        mask_curve = cache['_brain_curve']
    else:
        mask_curve = _brain_curve(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    
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
    tumor_mask = obj_segments[:,:,0]
    tumor_mask = obj_segments[:,:,1]

    if '_brain_curve' in cache:
        mask_curve = cache['_brain_curve']
    else:
        mask_curve = _brain_curve(obj_segments, h_pix_size, w_pix_size, brain_boundary_pts, cache)
    
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
    # '../data/LB-Dual GFAP CK IHC/LB01.svs',
    # '../data/LB-Dual GFAP CK IHC/LB02.svs',
    # '../data/LB-Dual GFAP CK IHC/LB03.svs',
    # '../data/LB-Dual GFAP CK IHC/LB04.svs',
    # '../data/LB-Dual GFAP CK IHC/LB05.svs',
    # '../data/LB-Dual GFAP CK IHC/LB06.svs',
    # '../data/LB-Dual GFAP CK IHC/LB07.svs',
    # '../data/LB-Dual GFAP CK IHC/LB08.svs',
    # '../data/LB-Dual GFAP CK IHC/LB09.svs',
    # '../data/LB-Dual GFAP CK IHC/LB10.svs',
    # '../data/LB-Dual GFAP CK IHC/LB13.svs',
    # '../data/LB-Dual GFAP CK IHC/LB14.svs',
    # '../data/LB-Dual GFAP CK IHC/LB15.svs',
    # '../data/LB-Dual GFAP CK IHC/LB16.svs',
    # '../data/LB-Dual GFAP CK IHC/LB17.svs',
    # '../data/LB-Dual GFAP CK IHC/LB18.svs',
    # '../data/LB-Dual GFAP CK IHC/LB19.svs',
    # '../data/LB-Dual GFAP CK IHC/LB20.svs',
    # '../data/LB-Dual GFAP CK IHC/LB21.svs',
    # '../data/LB-Dual GFAP CK IHC/LB22.svs',
    # # '../data/LB-Dual GFAP CK IHC/LB24.svs', ### Necrotic, discard
    # '../data/LB-Dual GFAP CK IHC/LB31.svs',
    # '../data/LB-Dual GFAP CK IHC/LB34.svs',
    # '../data/LB-Dual GFAP CK IHC/LB35.svs',
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
                _brain_curve,
                _brain_perimeter,
                whole_brain_area,
                whole_tumor_area,
                _brain_convex_hull,
                _brain_largest_area,
                invasion_count,
                delaunay_score,
                tumor_components,
                interface_ellipse,
                cooccurrence_mat,
                # tiled_count,
                # tiled_sum,
                brain_compactness,
                brain_convexity,
                brain_solidity,
                convex_overlap,
                # filled_overlap,
                # enclosing_circle_overlap,
                # _mask_curvature,
                # mask_area,
                # #circular_variance,
                # bending_energy,
                # total_abs_curv,
                # nearest_boundary_score,
                # surroundedness_degree,
            ],
            brain_boundary_pts
        )

        features_summary = all_features.describe(percentiles = np.arange(.25, 1, .25), include = 'all')
        features_summary.index.name = 'summary'
        features_summary.reset_index(inplace=True)
        features_summary = features_summary >> mutate(case = int(src[-6:-4]))
        
        all_features.hist(bins=30, grid=False, xlabelsize=5)
        plt.savefig('steps/%s_features.png'%pref); plt.close() # plt.show()
        all_features.to_csv('steps/%s_features.csv'%pref, index=False)

        if plates_info is None:
            plates_info = features_summary
        else:
            plates_info = pd.concat([plates_info, features_summary])
            plates_info.to_csv('plates_info_dual.temp.csv', index=False)
        
    print('Done\n')

print(plates_info)
plates_info.to_csv('plates_info_dual.csv', index=False)
