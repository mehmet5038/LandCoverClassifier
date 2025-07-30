import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from skimage.feature import graycomatrix, graycoprops
from sklearn.decomposition import PCA

import joblib


def extract_features(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # HOG
    hog_feat = hog(image_gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)

    # LBP
    lbp = local_binary_pattern(image_gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)

    # HSV histogram
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
    hsv_hist = np.concatenate([hist_h, hist_s, hist_v])

    # GLCM
    glcm = graycomatrix(image_gray, [1], [0], levels=256, symmetric=True, normed=True)
    glcm_feat = [graycoprops(glcm, prop).flatten()[0] for prop in ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation')]

    return np.concatenate([hog_feat, lbp_hist, hsv_hist, glcm_feat])


def apply_pca(isSkyView, features, n_components=100):
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    if isSkyView:
        joblib.dump(pca, "../../models/ml_features/skyview_pca.pkl")
    else:
        joblib.dump(pca, "../../models/ml_features/uc_merced_pca.pkl")
    return reduced_features
