import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from lesson_functions import bin_spatial, color_hist, get_hog_features

# Classifier data
svc_data = None


#
# Loads and initialize HOG+SVM classifier data from pickle file.
# Returns classifier data.
#
def load_classifier(pickle_file="HOGClassifier.p"):
    global svc_data
    dist_pickle = pickle.load( open(pickle_file, "rb" ) )
    svc_data = {}
    svc_data['svc'] = dist_pickle['svc']
    svc_data['X_scaler'] = dist_pickle['scaler']

    svc_data['color_space']    = dist_pickle['color_space']
    svc_data['spatial_size']   = dist_pickle['spatial_size']
    svc_data['hist_bins']      = dist_pickle['hist_bins']
    svc_data['orient']         = dist_pickle['orient']
    svc_data['pix_per_cell']   = dist_pickle['pix_per_cell']
    svc_data['cell_per_block'] = dist_pickle['cell_per_block']
    svc_data['hog_channel']    = dist_pickle['hog_channel']
    svc_data['spatial_feat']   = dist_pickle['spatial_feat']
    svc_data['hist_feat']      = dist_pickle['hist_feat']
    svc_data['hog_feat']       = dist_pickle['hog_feat']
    return svc_data


def convert_rgb_color(img, conv='YCrCb'):
    if conv == 'RGB':
        return np.copy(img)
    if conv == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conv == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)


#
# Define a single function that can extract features using hog sub-sampling and make predictions.
# The region: (0, ystart) to (image_width, ystop). 
# Params: img - source float32 RGB image with normed channels in range [0..1].
#         ystart - top Y of region
#         ystop - bottom Y of region
#         scale - scale of searching window.
# Returns: list of rectangles [(X1,Y1), (X2,Y2)].
#
def find_cars(img, ystart, ystop, scale, svc, X_scaler, params):
    
    color_space    = params['color_space']
    spatial_size   = params['spatial_size']
    hist_bins      = params['hist_bins']
    orient         = params['orient']
    pix_per_cell   = params['pix_per_cell']
    cell_per_block = params['cell_per_block']
    hog_channel    = params['hog_channel']
    
    spatial_feat   = params['spatial_feat']
    hist_feat      = params['hist_feat']
    hog_feat       = params['hog_feat']

    assert(hog_channel == 'ALL')
    assert(spatial_feat == True)
    assert(hist_feat == True)
    assert(hog_feat == True)
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_rgb_color(img_tosearch, conv=color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    bboxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                box = [(xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart)]
                bboxes.append(box)
                
    return bboxes
    

# Function detects vehicles in frame RGB image.
# Params: img - uint8 RGB image.
# Returns: list of windows [(x1,y1), (X2,Y2)] around detected vehicles.
def detect_vehicles(img, verbose=False):
    global svc_data
    
    ystart = 400
    ystop = 656
#    scale = 1.5
#    scale = 2
    scale = 1.2

    svc = svc_data['svc']
    X_scaler = svc_data['X_scaler']

    if verbose:
        print(svc_data)

    img = img.astype(np.float32)/255

    bboxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, svc_data)
    
    return bboxes
    
