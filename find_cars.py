import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label
from lesson_functions import bin_spatial, color_hist, get_hog_features, draw_boxes
from collections import deque

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




# Returns heatmap for list of bounding boxes.
def heatmap_from_detections(img, bbox_list):
    h,w,_ = img.shape
    heatmap = np.zeros((h,w)).astype(np.float32)
    
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        x1,y1 = box[0]
        x2,y2 = box[1]
        heatmap[y1:y2, x1:x2] += 1

     # Return updated heatmap
    return heatmap# Iterate through list of bboxes
       
#
# Applies threshold to heatmap.
# Params: heatmap - float32 image with one chanel.
#         threshold - all pixels <= threshold set to zero.
# Returns uint8 gray image
#
def apply_threshold(heatmap, threshold):
    # create a copy to exclude modification of input heatmap
    heatmap = np.copy(heatmap)
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    heatmap = np.clip(heatmap, 0, 255)
    # Return thresholded map
    return heatmap
    

def draw_labeled_bboxes(img, labels, color=(0, 0, 255), thick=6):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)
    # Return the image
    return img
    

#
# Returns list of bounding boxes for detected labes (cars) where every
# bounding box [(X1,Y1), (X2,Y2)].
#
def get_labeled_bboxes(labels):
    bboxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))        
        bboxes.append(bbox)
    # Return list of bounding boxes
    return bboxes

    
    
#
# Define a single function that can extract features using hog sub-sampling and make predictions.
# The region: (0, ystart) to (image_width, ystop). 
# Params: img - source float32 RGB image with normed channels in range [0..1].
#         ystart - top Y of region
#         ystop - bottom Y of region
#         scale - scale of searching window.
#         cells_per_step - instead of overlap, define how many cells to step.
# Returns: list of bouding boxes [(X1,Y1), (X2,Y2)].
#
def find_cars(img, ystart, ystop, scale, svc, X_scaler, params, cells_per_step = 2):
    
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
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    bboxes = []
    for xb in range(nxsteps+1):
        for yb in range(nysteps+1):
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


    
#
# Generates grid used for car detection for visualizarion.
# The region: (0, ystart) to (image_width, ystop). 
# Params: img - source float32 RGB image with normed channels in range [0..1].
#         ystart - top Y of region
#         ystop - bottom Y of region
#         scale - scale of searching window.
#         cells_per_step - instead of overlap, define how many cells to step
# Returns: image.
#
def find_cars_grid(img, ystart, ystop, scale, cells_per_step=8):
    
    pix_per_cell   = 8
    
    roi_width = img.shape[1]
    roi_height = ystop - ystart;
    
    if scale != 1:
        roi_width = np.int(roi_width / scale)
        roi_height = np.int(roi_height / scale)
        
    # Define blocks and steps as above
    nxblocks = (roi_width // pix_per_cell)-1
    nyblocks = (roi_height // pix_per_cell)-1 
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        
    bboxes = []
    for xb in range(nxsteps+1):
        for yb in range(nysteps+1):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            
            xbox_left = np.int(xleft*scale)
            ytop_draw = np.int(ytop*scale)
            win_draw = np.int(window*scale)
            box = [(xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart)]
            bboxes.append(box)
    
    return bboxes


ystart_0 = 400-2*8
ystop_0 = 400+1*64+2*8
scale_0 = 0.8

def find_cars_grid_0(img, ystart=ystart_0, ystop=ystop_0, scale=scale_0, cells_per_step=8):
    return find_cars_grid(img, ystart, ystop, scale, cells_per_step)

    
ystart_1 = 400-2*8
ystop_1 = 400+2*64+2*8
scale_1 = 1.3

def find_cars_grid_1(img, ystart=ystart_1, ystop=ystop_1, scale=scale_1, cells_per_step=8):
    return find_cars_grid(img, ystart, ystop, scale, cells_per_step)


ystart_2 = 400-2*8
#ystop_2 = 400+2*64+2*8
ystop_2 = 656
scale_2 = 1.8

def find_cars_grid_2(img, ystart=ystart_2, ystop=ystop_2, scale=scale_2, cells_per_step=8):
    return find_cars_grid(img, ystart, ystop, scale, cells_per_step)


#ystart_3 = 400-2*8
#ystop_3 = 656
#scale_3 = 2
ystart_3 = 400-32
ystop_3 = 656
scale_3 = 2.3

def find_cars_grid_3(img, ystart=ystart_3, ystop=ystop_3, scale=scale_3, cells_per_step=8):
    return find_cars_grid(img, ystart, ystop, scale, cells_per_step)

    
def find_cars_grid_160(img, ystart=400, ystop=656, scale=2.5, cells_per_step=8):
    return find_cars_grid(img, ystart, ystop, scale, cells_per_step)

    
#
# Function detects vehicles in frame RGB image.
# Params: img - uint8 RGB image.
# Returns: list of bounding boxes [(x1,y1), (X2,Y2)] where classifier reported positive detections.
#
def find_cars_multiscale(img, scales=[0,1,2,3], verbose=False):
    global svc_data
    
    global ystart_1, ystart_2, ystart_3
    global ystop_1, ystop_2, ystop_3
    global scale_1, scale_2, scale_3
    
#    ystart = 400
#    ystop = 656
#    scale = 1.5
#    scale = 2
#    scale = 1.2

    assert(np.max(img) <= 1)

    svc = svc_data['svc']
    X_scaler = svc_data['X_scaler']

    if verbose:
        print(svc_data)

    bboxes = []
    
    # Scale 0
    if 0 in scales:
        boxes = find_cars(img, ystart_0, ystop_0, scale_0, svc, X_scaler, svc_data)
        if len(boxes):
            bboxes.extend(boxes)

    # Scale 1
    if 1 in scales:
        boxes = find_cars(img, ystart_1, ystop_1, scale_1, svc, X_scaler, svc_data)
        if len(boxes):
            bboxes.extend(boxes)

    # Scale 2
    if 2 in scales:
        boxes = find_cars(img, ystart_2, ystop_2, scale_2, svc, X_scaler, svc_data)
        if len(boxes):
            bboxes.extend(boxes)

    # Scale 3
    if 3 in scales:
        boxes = find_cars(img, ystart_3, ystop_3, scale_3, svc, X_scaler, svc_data)
        if len(boxes):
            bboxes.extend(boxes)
            
#    scale = 2
#    boxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, svc_data)
#    if len(boxes):
#        bboxes.extend(boxes)

    return bboxes
    

# The class receives the paramenets of bounding box detection and averages box detection.
class BoundingBoxes:
    def __init__(self, nf = 5):
        # defines the length of queue used to buffer data from 'nf' frames
        self.nf = nf
        # hot windows of the last n frames
        self.recent_boxes = deque([], maxlen=nf)
        # hot windows of current frame
        self.currect_boxes = None
        # all hot windows for last n frames
        self.all_boxes = []
    
    def update_all_boxes_(self):
        all_boxes = []
        for boxes in self.recent_boxes:
            all_boxes += boxes
        if len(all_boxes) == 0:
            self.all_boxes = []
        else:
            self.all_boxes = all_boxes
    
    def add(self, boxes):
        self.currect_boxes = boxes
        self.recent_boxes.appendleft(boxes)
        self.update_all_boxes_()
 


#
# Detects vehicles in the frame.
# Params: image - uint8 RGB image.
#         frame_idx - frame index.
#         verbose - on/off debug info.
# Returns: bounding boxed with detected vehicles.
#          In verbose mode returns additional params: hot_windows, heatmap, label
#
def detect_vehicles(image, frame_idx, scales=[0,1,2,3], avgBoxes=None, thresh=1, useHeatmap=True, verbose=False):
    
    hot_windows = find_cars_multiscale(image, scales=scales)
    
    if avgBoxes:
        avgBoxes.add(hot_windows)
        hot_windows = avgBoxes.all_boxes
    
    heatmap=[]
    labels=[]
        
    if useHeatmap:
        heatmap = heatmap_from_detections(image, hot_windows)
        heatmap_thresh = apply_threshold(heatmap, thresh)
        
        labels = label(heatmap_thresh)
        bboxes = get_labeled_bboxes(labels)
    else:
        bboxes = hot_windows
    
    if verbose == False:
        return bboxes
        
    return bboxes, hot_windows, heatmap, labels
    
    

def draw_cars_grids(img, scales=[0,1,2,3], cell_blocks=True, color=(0, 0, 255), thick=4):

    draw_img = np.copy(img)
    
    if 3 in scales:
        if cell_blocks:
            bboxes = find_cars_grid_3(img, cells_per_step=2)
            draw_img = draw_boxes(draw_img, bboxes, color=(255, 0, 0), thick=2)
        bboxes = find_cars_grid_3(img)
        draw_img = draw_boxes(draw_img, bboxes, color=color, thick=thick)
        
    if 2 in scales:
        if cell_blocks:
            bboxes = find_cars_grid_2(img, cells_per_step=2)
            draw_img = draw_boxes(draw_img, bboxes, color=(255, 0, 0), thick=2)
        bboxes = find_cars_grid_2(img)
        draw_img = draw_boxes(draw_img, bboxes, color=color, thick=thick)        

    if 1 in scales:
        if cell_blocks:
            bboxes = find_cars_grid_1(img, cells_per_step=2)
            draw_img = draw_boxes(draw_img, bboxes, color=(255, 0, 0), thick=2)
        bboxes = find_cars_grid_1(img)
        draw_img = draw_boxes(draw_img, bboxes, color=color, thick=thick)
                    
    if 0 in scales:
        if cell_blocks:
            bboxes = find_cars_grid_0(img, cells_per_step=2)
            draw_img = draw_boxes(draw_img, bboxes, color=(255, 0, 0), thick=2)    
        bboxes = find_cars_grid_0(img)
        draw_img = draw_boxes(draw_img, bboxes, color=color, thick=thick)       
            
    return draw_img

    
# Draw debug board with Binarization-View, Lane-Detesction-View
def draw_debug_board(img, frame_ind, bboxes, hot_windows, heatmap, labels):
    
    # prepare RGB heatmap image from float32 heatmap channel
    img_heatmap = (np.copy(heatmap) / np.max(heatmap) * 255.).astype(np.uint8);
    img_heatmap = cv2.applyColorMap(img_heatmap, colormap=cv2.COLORMAP_HOT)
    img_heatmap = cv2.cvtColor(img_heatmap, cv2.COLOR_BGR2RGB)

    # prepare RGB labels image from float32 labels channel
    img_labels = (np.copy(labels) / np.max(labels) * 255.).astype(np.uint8);
    img_labels = cv2.applyColorMap(img_labels, colormap=cv2.COLORMAP_HOT)
    img_labels = cv2.cvtColor(img_labels, cv2.COLOR_BGR2RGB)
    
    # draw hot_windows in the frame
    img_hot_windows = np.copy(img)
    img_hot_windows = draw_boxes(img_hot_windows, hot_windows, thick=2)
    
    ymax = 0
    
    board_x = 5
    board_y = 5
    board_ratio = (img.shape[0] - 3*board_x)//3 / img.shape[0] #0.25
    board_h = int(img.shape[0] * board_ratio)
    board_w = int(img.shape[1] * board_ratio)
        
    ymin = board_y
    ymax = board_h + board_y
    xmin = board_x
    xmax = board_x + board_w

    offset_x = board_x + board_w

    # draw hot_windows in the frame
    img_hot_windows = cv2.resize(img_hot_windows, dsize=(board_w, board_h), interpolation=cv2.INTER_LINEAR)
    img[ymin:ymax, xmin:xmax, :] = img_hot_windows
    
    # draw heatmap in the frame
    xmin += offset_x
    xmax += offset_x
    img_heatmap = cv2.resize(img_heatmap, dsize=(board_w, board_h), interpolation=cv2.INTER_LINEAR)
    img[ymin:ymax, xmin:xmax, :] = img_heatmap
    
    # draw heatmap in the frame
    xmin += offset_x
    xmax += offset_x
    img_labels = cv2.resize(img_labels, dsize=(board_w, board_h), interpolation=cv2.INTER_LINEAR)
    img[ymin:ymax, xmin:xmax, :] = img_labels
    
    return img;
    
   
#
# Detects and lables vehicles in the frame.
# Params: img - RGB24 frame.
#         frame_ind - frame index.
#         useHeatmap - use heatmap for vehicle detection.
#         thresh - threshold for heatmap
#         avgBoxes - averaging hot windows along several frames.
# Returns: frame with detected vehicles
#
def process_image_hog_pipeline(image, frame_ind, useHeatmap=True, thresh=4, avgBoxes=None, verbose=False, verboseSaveHeatmaps=False):
    
    out_dir_model       ='./models/model/'
    out_dir_heatmap     = out_dir_model + 'heatmap/'
    out_dir_heatmap_val = out_dir_model + 'heatmap_val/'
    out_dir_labels      = out_dir_model + 'labels/'
    
    frame_ind += 1
        
    draw_image = np.copy(image)
    image = image.astype(np.float32)/255

    if verbose == False:
        bboxes = detect_vehicles(image, frame_ind, scales=[0,1,2,3], thresh=thresh, useHeatmap=useHeatmap, avgBoxes=avgBoxes)
        result = draw_boxes(draw_image, bboxes, thick=2)
    else:
        bboxes, hot_windows, heatmap, labels = detect_vehicles(image, frame_ind, scales=[0,1,2,3], thresh=thresh,
                                                               useHeatmap=useHeatmap, avgBoxes=avgBoxes, verbose=True)
        result = draw_boxes(draw_image, bboxes, thick=2)    
        result = draw_debug_board(result, frame_ind, bboxes, hot_windows, heatmap, labels[0])
        
        if verboseSaveHeatmaps:
            # save heatmap as hit image
            f_path = out_dir_heatmap + str(frame_ind) + '.png'
            plt.imsave(f_path, heatmap, cmap='hot')
            # save heatmap values
            heatmap_val = (np.dstack((heatmap,heatmap,heatmap))).astype(np.uint8)
            f_path = out_dir_heatmap_val + str(frame_ind) + '.png'
            mpimg.imsave(f_path, heatmap_val)
            # save labels
            f_path = out_dir_labels + str(frame_ind) + '.png'
            gray = (np.copy(labels[0])*255/np.max(labels[0])).astype(np.uint8)
            gray = np.dstack((gray,gray,gray))
            mpimg.imsave(f_path, gray)
    
    
        # add frame_index text at the bottom of board
        xmax = 900
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, 'frame {:d}'.format(frame_ind), (xmax + 20, 50), font, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
    
    return result