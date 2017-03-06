#import time
from find_cars import load_classifier, BoundingBoxes, process_image_hog_pipeline

#import imageio
#imageio.plugins.ffmpeg.download()
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip


frame_ind = 0
avgBoxes = BoundingBoxes(10)
verbose = True


# Tuning & saving heatmaps
def process_image_measure_mode(image):
    global frame_ind    
    frame_ind += 1
    result = process_image_hog_pipeline(image, frame_ind, thresh=1, verbose=True, verboseSaveHeatmaps=True)      
    return result
    
    
# Generate final version
def process_image(image):
    global frame_ind
    global avgBoxes
    global verbose
    
    frame_ind += 1
    
    # Set verbose to False to hide top debug bar.
    result = process_image_hog_pipeline(image, frame_ind, thresh=29, avgBoxes=avgBoxes, verbose=verbose)      
    return result
    

if __name__ == '__main__':
    
    MeasureMode = False
    
    out_dir='./output_images/'
    pickle_file = "HOGClassifier.p"
        
    load_classifier(pickle_file)
    
    output = out_dir + 'processed_project_video.mp4'
    clip = VideoFileClip("project_video.mp4")
    
    #output = out_dir + 'processed_test_video.mp4'
    #clip = VideoFileClip("test_video.mp4")
    
    if MeasureMode == False:
        out_clip = clip.fl_image(process_image)
    else:
        out_clip = clip.fl_image(process_image_measure_mode)
        
    out_clip.write_videofile(output, audio=False)

