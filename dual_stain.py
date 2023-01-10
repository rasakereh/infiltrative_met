import slideio
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn.decomposition import NMF, PCA
from sklearn.cluster import KMeans

import histomicstk as htk
import skimage.io
import skimage.measure
import skimage.color

EPSILON = 1e-5

def read_slide(src, rect=None, size=None):
    brain_slide = slideio.open_slide(src, 'SVS')
    scene = brain_slide.get_scene(0)
    img_size = scene.size
    if rect is None:
        x0, y0 = randint(int(img_size[0]*.1), int(img_size[0]*.4)), randint(int(img_size[1]*.1), int(img_size[1]*.4))
        x1, y1 = randint(int(img_size[0]*.6), int(img_size[0]*.9)), randint(int(img_size[1]*.6), int(img_size[1]*.9))
        rect = (x0, y0, x1, y1)
    if size is None:
        width = min(9000, max(5000, rect[2] - rect[0]))
        size = (width, 0)
    print("img_size:", img_size)
    image = scene.read_block(rect, size)
    cropped_w = (int(image.shape[1] * .01), int(image.shape[1] * .8))
    cropped_h = (int(image.shape[0] * .01), int(image.shape[0] * .8))
    print("==============", image.shape)
    image = image[cropped_h[0]:cropped_h[1], cropped_w[0]:cropped_w[1], 0:3]
    print("==============", image.shape)
    
    return image


def color_deconvolution_NMF(image, color_cnt=3):
    color_samples = image.reshape(-1, 3)
    model = NMF(n_components=color_cnt, init='custom', verbose=1, tol=0.001, max_iter=100)
    W = model.fit_transform(
        color_samples,
        H = np.array([
            [45., 11, 8], #brown
            [191, 65, 84], #pink
            [133, 130, 178], #violet
        ]),
        W = np.ones(color_samples.shape) / 3
    )
    H = model.components_
    # W, H, info = NMF_ANLS_BLOCKPIVOT().run(color_samples, 3)

    print(color_samples.shape, W.shape, H.shape)
    W = W.reshape((image.shape[0], image.shape[1], color_cnt))
    W /= np.amax(W)

    return W, H

def color_deconvolution_HTK(image, color_cnt=3, sparsity_factor = 0.5):
    I_0 = 255
    stain_colors = np.array([
        [45., 11, 8], #brown
        [191, 65, 84], #pink
        [133, 130, 178], #violet
    ]).T / 255
    # im_sda = htk.preprocessing.color_conversion.rgb_to_sda(image, I_0)
    # w_est = htk.preprocessing.color_deconvolution.separate_stains_xu_snmf(
    #     im_sda, stain_colors, sparsity_factor,
    # )

    # # perform sparse color deconvolution
    # imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(
    #     image,
    #     htk.preprocessing.color_deconvolution.complement_stain_matrix(w_est),
    #     I_0,
    # )

    # w_est = htk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(image, I_0)
    # # Perform color deconvolution
    # imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(image, w_est, I_0)

    # print('Estimated stain colors (in rows):', w_est.T, sep='\n')

    # perform standard color deconvolution
    imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(image, stain_colors)

    return imDeconvolved.Stains

def color_finding(image):
    main_colors = np.array([
        [45., 11, 8], #brown
        [191, 65, 84], #pink
        [133, 130, 178], #violet
        [200, 150, 150], #pale brown
        [255, 255, 255], #background
    ])
    pixels = image.reshape((-1, 3))
    dists = np.hstack([
        np.linalg.norm(pixels - main_colors[j], axis=1).reshape(-1, 1) for j in range(main_colors.shape[0])
    ])

    min_mask = dists.argmin(axis=-1)[...,None] == np.arange(dists.shape[-1])
    weights = np.ones(dists.shape)
    weights[min_mask] = 0
    weights[~min_mask] = 1
    # weighted_dists = dists * weights + EPSILON
    dists = weights# * dists / np.sum(dists * weights, axis=1).reshape(-1, 1)
    dists = 1 - dists

    # print('max:', np.amax(np.sum(dists, axis=1)))

    return dists.reshape((image.shape[0], image.shape[1], main_colors.shape[0]))

def color_analysis(image):
    YCC_img = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
    LAB_img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    colors = image.reshape((-1, 3)) / 255.
    pixels = LAB_img.reshape((-1, 3))
    print(np.amin(pixels, axis=0), np.amax(pixels, axis=0))
    pca = PCA(n_components=2)
    
    chosen_index = np.random.randint(pixels.shape[0], size=30000)

    low_cols = pca.fit_transform(pixels)
    print(low_cols)
    plt.scatter(low_cols[chosen_index,0], low_cols[chosen_index,1], c=colors[chosen_index,:], alpha=0.75)
    plt.show()

def color_quantization(image, color_cnt=8):
    positions = np.array([(i/image.shape[0], j/image.shape[1]) for i in range(image.shape[0]) for j in range(image.shape[1])])
    final_points = np.hstack([image.reshape((-1, 3)) / 255., positions])

    kmeans = KMeans(n_clusters=color_cnt, random_state=0).fit(final_points)
    centers = np.uint8(kmeans.cluster_centers_[:,:3] * 255)
    res = centers[kmeans.labels_.flatten()]
    res = res.reshape(image.shape)

    return res

def YCbCr_quantization(image, color_cnt=8):
    # YCC_img = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB).reshape((-1, 3))
    LAB_img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).reshape((-1, 3))

    kmeans = KMeans(n_clusters=color_cnt, random_state=0).fit(LAB_img)
    centers = np.uint8(kmeans.cluster_centers_)
    print(centers)
    centers = np.uint8(np.array([
        [0, 110, 84],
        [0, 110, 172],
        [0, 196, 84],
        [0, 196, 172],
        [255, 110, 84],
        [255, 110, 172],
        [255, 196, 84],
        [255, 196, 172]
    ]))
    print(centers)
    res = centers[kmeans.labels_.flatten()]
    res = res.reshape(image.shape)
    # res = cv2.cvtColor(res, cv2.COLOR_YCR_CB2RGB)
    res = cv2.cvtColor(res, cv2.COLOR_LAB2RGB)

    return res


src_list = [
    '../data/LB-Dual GFAP CK IHC/LB01.svs',
    # '../data/LB-Dual GFAP CK IHC/LB02.svs',
    # '../data/LB-Dual GFAP CK IHC/LB03.svs',
    # '../data/LB-Dual GFAP CK IHC/LB04.svs',
    # '../data/LB-Dual GFAP CK IHC/LB05.svs',
    # '../data/LB-Dual GFAP CK IHC/LB07.svs',
    # '../data/LB-Dual GFAP CK IHC/LB09.svs',
    # '../data/LB-Dual GFAP CK IHC/LB36.svs',
    # '../data/LB-Dual GFAP CK IHC/LB37.svs'
]

stain_colors = np.array([
    [45., 11, 8], #brown
    [191, 65, 84], #pink
    [133, 130, 178], #violet
]).T

for src in src_list:
    print('File: %s...'%src)
    pref = src.split('/')[3].split('.')[0]
    for i in range(1):
        print('Loading instance %d...'%(i+1))
        curr_img = read_slide(src)
        
        ########################
        # Perform kmeans color segmentation, grayscale, Otsu's threshold
        print('Color quantization ...')
        channels = YCbCr_quantization(curr_img)

        _, axs = plt.subplots(2)
        axs[0].imshow(curr_img)
        axs[0].set_title('curr_img')
        axs[1].imshow(channels)
        axs[1].set_title('quantized')
        plt.show()


        # curr_img, Inorm, H, E = normalize_hist(curr_img)
        # channels = color_deconvolution_HTK(curr_img)

        # channels = color_finding(curr_img)

        # _, axs = plt.subplots(2)
        # axs[0].imshow(curr_img)
        # axs[0].set_title('curr_img')
        # axs[1].imshow(channels[:,:,:3])
        # axs[1].set_title('Segmented')
        # # axs[1,0].imshow(channels[:,:,1], cmap='gray')
        # # axs[1,0].set_title('Pink')
        # # axs[1,1].imshow(channels[:,:,2], cmap='gray')
        # # axs[1,1].set_title('Violet')
        # plt.show()

        # print('Color analysis...')
        # color_analysis(curr_img)

        
    print('Done\n')
