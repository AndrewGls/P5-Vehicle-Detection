import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from lesson_functions import single_img_features
from sklearn.utils import shuffle


# Returns two lists: cars and notcars list of image files.
def load_data_sets():
    cars = glob.glob('./vehicles/GTI_Far/*.png')
    cars += glob.glob('./vehicles/GTI_MiddleClose/*.png')
    cars += glob.glob('./vehicles/GTI_Left/*.png')
    cars += glob.glob('./vehicles/GTI_Right/*.png')
    cars += glob.glob('./vehicles/KITTI_extracted/*.png')
    notcars = glob.glob('./non-vehicles/Extras/*.png')
    notcars += glob.glob('./non-vehicles/GTI/*.png')

    return cars, notcars
    

# Saves cars and notcars lists into pickle file.
def save_data_sets(pickle_file, cars, notcars):    
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                {
                    'cars': cars,
                    'notcars': notcars,
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
        
# Loads cars and notcars sets from pickle file.
def load_cars_norcars(data_file='data.p'):
    with open(data_file, mode='rb') as f:
        data = pickle.load(f)
    cars = data['cars']
    notcars = data['notcars']    
    return cars, notcars
    
    
    
# Save the data for easy access
def save_classifier_data(pickle_file, X_train, y_train, X_test, y_test):
    print('Saving features/labels to pickle file...')
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'y_test': y_test                
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save classifier data to', pickle_file, ':', e)
        raise
    print('Classifier data cached in pickle file.')
    

# Save HOG Classifier to pickle file.
def save_classifier(pickle_file, svc, X_scaler, params):
    print('Saving classiifier to pickle file...')
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                {   'svc':svc, 
                    'scaler': X_scaler,

                    'color_space': params['color_space'],
                    'orient': params['orient'],
                    'pix_per_cell': params['pix_per_cell'],
                    'cell_per_block': params['cell_per_block'],
                    'hog_channel': params['hog_channel'],
                    'spatial_size': params['spatial_size'],
                    'hist_bins': params['hist_bins'],
                    'spatial_feat': params['spatial_feat'],
                    'hist_feat': params['hist_feat'],
                    'hog_feat': params['hog_feat']
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save classifier to', pickle_file, ':', e)
        raise
    print('Classifier saved in pickle file.')
    
    
    
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, params):
                     
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
    
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        img = mpimg.imread(file)
        
        img_features = single_img_features(img, color_space=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    
        features.append(img_features)
        
    # Return list of feature vectors
    return features


    
if __name__ == '__main__':
    
    # Loads cars and notcars image files
    cars, notcars = load_data_sets()
    
    print('Number of samples in cars set: ', len(cars))
    print('Number of samples in notcars set: ', len(notcars))
    
    
    # Extracts features from image files
    
    params = {}
    
#    params['color_space'] = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
#    params['orient'] = 9  # HOG orientations
#    params['pix_per_cell'] = 8 # HOG pixels per cell
#    params['cell_per_block'] = 2 # HOG cells per block
#    params['hog_channel'] = 'ALL' # Can be 0, 1, 2, or "ALL"
#    params['spatial_size'] = (32, 32) # Spatial binning dimensions
#    params['hist_bins'] = 32    # Number of histogram bins
#    params['spatial_feat'] = True # Spatial features on or off
#    params['hist_feat'] = True # Histogram features on or off
#    params['hog_feat'] = True # HOG features on or off

    params['color_space'] = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    params['orient'] = 9  # HOG orientations
    params['pix_per_cell'] = 8 # HOG pixels per cell
    params['cell_per_block'] = 2 # HOG cells per block
    params['hog_channel'] = 'ALL' # Can be 0, 1, 2, or "ALL"
    params['spatial_size'] = (16, 16) # Spatial binning dimensions
    params['hist_bins'] = 16    # Number of histogram bins
    params['spatial_feat'] = True # Spatial features on or off
    params['hist_feat'] = True # Histogram features on or off
    params['hog_feat'] = True # HOG features on or off
    
    t1=time.time()
    
    cars_feats = extract_features(cars, params)
    notcars_feats = extract_features(notcars, params)
    
    t2 = time.time()
    print(round(t2-t1, 2), 'second to extract features (HOG,spatial and color features).')    
    assert(len(cars_feats) == len(cars))
    assert(len(notcars_feats) == len(notcars))
    
    
    # Combine and Normalize Features

    # Create an array stack of feature vectors
    X = np.vstack((cars_feats, notcars_feats)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    # Define the labels vector
    y = np.hstack((np.ones(len(cars)), np.zeros(len(notcars))))
    
    
    # Split up data into randomized training and test sets
    
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)    
    print('Number of samples in train set: ', len(X_train))
    print('Number of samples in test set: ', len(X_test))

    
    # Save prepared train and test sets to pickle file
    
    # Uncomment to save train and test sets to disck.
    #save_classifier_data('HOGClassifierData.p', X_train, y_train, X_test, y_test)
    
    
    # Train SVM Classifier

    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 100
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    
    # Save HOG Classifier to file
    save_classifier('HOGClassifier.p', svc, X_scaler, params)
    
    
    # Test prediction

    from find_cars import find_cars
    from lesson_functions import draw_boxes
    
    dist_pickle = pickle.load( open("HOGClassifier.p", "rb" ) )
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    
    params = {}
    params['color_space']    = dist_pickle['color_space']
    params['orient']         = dist_pickle['orient']
    params['pix_per_cell']   = dist_pickle['pix_per_cell']
    params['cell_per_block'] = dist_pickle['cell_per_block']
    params['hog_channel']    = dist_pickle['hog_channel']
    params['spatial_size']   = dist_pickle['spatial_size']
    params['hist_bins']      = dist_pickle['hist_bins']
    params['spatial_feat']   = dist_pickle['spatial_feat']
    params['hist_feat']      = dist_pickle['hist_feat']
    params['hog_feat']       = dist_pickle['hog_feat']
    
    print('color_space: ', params['color_space'])
    print('orient: ', params['orient'])
    print('pix_per_cell: ', params['pix_per_cell'])
    print('cell_per_block: ', params['cell_per_block'])
    print('hog_channel: ', params['hog_channel'])
    print('spatial_size: ', params['spatial_size'])
    print('hist_bins: ', params['hist_bins'])
    print('spatial_feat: ', params['spatial_feat'])
    print('hist_feat: ', params['hist_feat'])
    print('hog_feat: ', params['hog_feat'])
    
    #img = mpimg.imread('test_images/test1.jpg')
    #img = mpimg.imread('test_images/test2.jpg')
    #img = mpimg.imread('test_images/test3.jpg')
    img = mpimg.imread('test_images/test4.jpg')
    #img = mpimg.imread('test_images/test5.jpg')
    #img = mpimg.imread('test_images/test6.jpg')
    
    ystart = 400
    ystop = 656
    scale = 1.5
        
    t=time.time()
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    bboxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, params)
    
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to detect using scale:', scale)
    
    draw_img = draw_boxes(draw_img, bboxes)
    
    plt.figure(figsize=(8,8))
    plt.xlim(0, 1280)
    plt.ylim(720, 0) 
    plt.imshow(draw_img)

    