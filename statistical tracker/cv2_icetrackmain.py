import logging
import logging.handlers
import os
import sys

import cv2
import numpy as np

from cv2_icetrackclass import ObjectTrackerClass
from skimage.filters import threshold_otsu

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TKAgg')
from skimage.color import rgb2gray
from skimage.util import invert

# ============================================================================

IMAGE_DIR = "images"
IMAGE_FILENAME_FORMAT = IMAGE_DIR + "/frame_%04d.png"

# Support either video file or individual frames
CAPTURE_FROM_VIDEO = False
if CAPTURE_FROM_VIDEO:
    IMAGE_SOURCE = "ice_video2.mp4" # Video file
else:
    IMAGE_SOURCE = IMAGE_FILENAME_FORMAT # Image sequence

# Time to wait between frames, 0=forever
WAIT_TIME = 1 # 250 # ms

LOG_TO_FILE = True

# Colours for drawing on processed frames    
BOUNDING_BOX_COLOUR = (255, 0, 0)
CENTROID_COLOUR = (0, 0, 255)

OBJECTS = None
MEANDRIFT = np.zeros(2,list)





####################        
#################### 




def init_logging():
    main_logger = logging.getLogger()

    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s] %(message)s'
        , datefmt='%Y-%m-%d %H:%M:%S')

    handler_stream = logging.StreamHandler(sys.stdout)
    handler_stream.setFormatter(formatter)
    main_logger.addHandler(handler_stream)

    if LOG_TO_FILE:
        handler_file = logging.handlers.RotatingFileHandler("debug.log"
            , maxBytes = 2**24
            , backupCount = 10)
        handler_file.setFormatter(formatter)
        main_logger.addHandler(handler_file)

    main_logger.setLevel(logging.DEBUG)

    return main_logger





####################        
#################### 




def save_frame(file_name_format, frame, label_format,frame_number=None):
    
    if frame_number != None:
        file_name = file_name_format % frame_number
        label = label_format % frame_number
    else:
        file_name = file_name_format
        label = label_format

    log.debug("Saving %s as '%s'", label, file_name)
    cv2.imwrite(file_name, frame)




####################        
#################### 




def get_centroid(x, y, w, h):
    
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1

    return (cx, cy)




####################        
#################### 




def DetectObject(binary_mask,shape, frame_number, frame):
    """
    detecting and filtering contours/objects in the frame

    Args:
        binary_mask ([np.ndarray]): binary image mask
        shape ([list(int)]): img dimensions
        frame_number ([int]): index of current frame
        frame ([np.ndarray]): current frame

    Returns:
        [list(lists)]: list of all foundings (bbox, centroid, frame_nr)
    """



    log = logging.getLogger("DetectObject")

    MIN_CONTOUR_WIDTH = 80
    MIN_CONTOUR_HEIGHT = 80
    MAX_CONTOUR_WIDTH = shape[1]/2
    MAX_CONTOUR_HEIGHT = shape[0]/2
    
    
    # Find the contours of any object/ice in the image
    contours, hierarchy = cv2.findContours(binary_mask
        , cv2.RETR_EXTERNAL
        , cv2.CHAIN_APPROX_SIMPLE)

    log.debug("Found %d objects contours.", len(contours))

    #list of valid objects in current frame
    foundings = []

    # check for contour thresholds
    for (i, contour) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        contour_valid = (w >= MIN_CONTOUR_WIDTH) and (h >= MIN_CONTOUR_HEIGHT) and (w <= MAX_CONTOUR_WIDTH) and (h <= MAX_CONTOUR_HEIGHT)

        if contour_valid == True:
            log.debug("Contour #%d: pos=(x=%d, y=%d) size=(w=%d, h=%d) valid=%s"
                , i, x, y, w, h, contour_valid)
            
            centroid = get_centroid(x, y, w, h)         #centroid of current contour
            foundings.append(((x, y, w, h), centroid, frame_number))

    return foundings





####################        
#################### 




def filter_mask(binary_mask):
    """
    refine binary mask

    Args:
        binary_mask ([np.ndarray]): binary image mask

    Returns:
        [np.ndarray]: refined binary mask
    """
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # remove small ice particles / shrink
    erosion = cv2.erode(binary_mask, kernel, iterations = 4)
    
    #resize ice space
    dilation = cv2.dilate(erosion, kernel, iterations = 2)
    
    
    return dilation




####################        
#################### 




def drift(OBJECTS):
    """
    class to calculate the MEANDRIFT of all known Objects
    using their ongoingly calculated mean drift

    Args:
        OBJECTS ([list(ObjectClass items)]): list of all known Objects so far
    """

    global MEANDRIFT

    drift_n = MEANDRIFT[0]
    index_n = MEANDRIFT[1]
    
    for Vehicle in OBJECTS:

        vector_n1 = np.array(Vehicle.MeanVector[0])
        offset = ( vector_n1 -  drift_n) / (index_n + 1)
        #update drift
        drift_n += offset
        index_n += 1
        
    MEANDRIFT[0] = drift_n
    MEANDRIFT[1] = index_n
    MEANDRIFT = MEANDRIFT[0]





####################        
#################### 





def ProcessFrame(frame_number, frame, ObjectCounter):
    """
    prepare frame, detect objects via binary contours
    call assignment class 

    Args:
        frame_number ([int]): index of current frame
        frame ([np.ndarray]): current frame
        ObjectCounter ([class]): Object assignment class

    Returns:
        [np.ndarray]: current frame with marked foundings
    """
    
    global OBJECTS 
    
    
    log = logging.getLogger("ProcessFrame")

    # Create a copy of source frame to draw into
    processed = frame.copy()

    # Remove the background
    #binary_mask = bg_subtractor.apply(frame, None, 0.01)
    
    #prepare treshold
    grey = rgb2gray(processed)
    otsu_thresh = threshold_otsu(grey)*1.1
    binary_mask = grey<otsu_thresh
    
    #binary version
    binary_mask = invert(binary_mask)
    binary_mask.dtype='uint8'
    #filter need uint8 type not bool
    binary_mask = filter_mask(binary_mask)
    
    #only binary contur version
    #binary_mask = binary_mask ^ ndimage.binary_dilation(binary_mask)
    
    save_frame(IMAGE_DIR + "/mask_%04d.png"
        , binary_mask*255, "foreground mask for frame #%d", frame_number)

    #find objects in frame
    foundings = DetectObject(binary_mask, frame.shape, frame_number, frame)

    #update and assign foundings and Objects
    log.debug("Updating object count in frame %i...", frame_number)
    OBJECTS = ObjectCounter.UpdateCount(foundings, frame_number, binary_mask, processed)

    #print all found and saved Objects on frame
    foundings = []
    for Obj in OBJECTS:
        if Obj.LastPosition[1] == frame_number:
            foundings.append((Obj.id, Obj.LastContour, Obj.LastPosition[0], Obj.LastPosition[1]))
    
    log.debug("Found %d valid vehicle contours.", len(foundings))
    for (i, found_ob) in enumerate(foundings):
        id, contour, centroid, frame_number = found_ob

        log.debug("Valid vehicle contour #%d: centroid=%s, bounding_box=%s", i, centroid, contour)

        x, y, w, h = contour

        # Mark the bounding box and the centroid on the processed frame
        cv2.rectangle(processed, (x, y), (x + w - 1, y + h - 1), BOUNDING_BOX_COLOUR, 1)
        cv2.circle(processed, centroid, 2, CENTROID_COLOUR, -1)
        cv2.putText(processed, str(id), fontFace = cv2.FONT_HERSHEY_SIMPLEX, color = (0,0,0), org = (centroid[0]+20, centroid[1]+20), fontScale = 3, thickness = 2)
    

    return processed




####################        
#################### 




def main():
    """
    loops over all frames, filteres the recognized Objects
    and saves them alll on two viduals
    """


    log = logging.getLogger("main")


    ## könnte hier einen background substractor bauen indem ich von einem anfangsbild die konturen (vom schiff)
    ## speichere und die nächsten male abziehe
    #log.debug("Creating background subtractor...")
    #bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    #log.debug("Pre-training the background subtractor...")
    #default_bg = cv2.imread(IMAGE_FILENAME_FORMAT % 33)
    #bg_subtractor.apply(default_bg, None, 1.0)



    ObjectCounter = None # Will be created after first frame is captured

    # Set up image source
    log.debug("Initializing video capture device #%s...", IMAGE_SOURCE)
    cap = cv2.VideoCapture(IMAGE_SOURCE)

    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    log.debug("Video capture frame size=(w=%d, h=%d)", frame_width, frame_height)



    log.debug("Starting capture loop...")
    frame_number = -1
    while True:
        frame_number += 1
        log.debug("Capturing frame #%d...", frame_number)
        ret, frame = cap.read()
        if not ret:
            log.error("Frame capture failed, stopping...")
            break

        log.debug("Got frame #%d: shape=%s", frame_number, frame.shape)

        if ObjectCounter is None:
            # We do this here, so that we can initialize with actual frame size
            log.debug("Creating vehicle counter...")
            ObjectCounter = ObjectTrackerClass(frame.shape[:2])


        # Archive raw frames from video to disk for later inspection/testing
        if CAPTURE_FROM_VIDEO:
            save_frame(IMAGE_FILENAME_FORMAT
                , frame, "source frame #%d", frame_number)

        #find and treat all Objects 
        log.debug("Processing frame #%d...", frame_number)
        processed = ProcessFrame(frame_number, frame, ObjectCounter)

        #archive frame with all valid foundings
        save_frame(IMAGE_DIR + "/processed_%04d.png"
            , processed, "processed frame #%d", frame_number)

        cv2.namedWindow('Processed Image',  cv2.WINDOW_KEEPRATIO)
        cv2.imshow('Processed Image', processed)
        log.debug("Frame #%d processed.", frame_number)
        c = cv2.waitKey(WAIT_TIME)
        if c == 27:
            log.debug("ESC detected, stopping...")
            break



    log.debug("Closing video capture device...")
    cap.release()
    cv2.destroyAllWindows()
    


    ####### filtering #######

    #filter by length
    OBJECTS[:] = [ o for o in OBJECTS
            if len(o.Positions)>=4]
    
    #filter by compare start-end difference with their MeanVector for coherence
    filtered_objects = []
    for Obj in OBJECTS:
        vector = (np.array(Obj.Positions[-1][0])-np.array(Obj.Positions[0][0]))/len(Obj.Positions)
        vector = (abs(vector[0]), vector[1])
        if (vector[1]<Obj.MeanVector[0][1]*1.4 and vector[1]>Obj.MeanVector[0][1]*.6) or (vector[0]<Obj.MeanVector[0][0]*1.4 and vector[0]>Obj.MeanVector[0][0]*.6):
            filtered_objects.append(Obj)

    drift(OBJECTS)

    #filter Objects by coherence to general Meandrift
    filter_more=[]
    for Obj in filtered_objects:
        if (Obj.MeanVector[0][1]<MEANDRIFT[0][1]*1.5 and Obj.MeanVector[0][1]>MEANDRIFT[0][1]*.5) or (Obj.MeanVector[0][0]<MEANDRIFT[0][0]*1.5 and Obj.MeanVector[0][0]>MEANDRIFT[0][0]*.5):
            filter_more.append(Obj)
    
    filtered_objects = filter_more
    del filter_more
    


    ####### save visuals #######

    #load first frame to mark all on it
    cap = cv2.VideoCapture(IMAGE_SOURCE)
    ret, frame = cap.read()
    frame2 = frame.copy()
    for (i, Obj) in enumerate(OBJECTS):
        obj_num = len(OBJECTS)
        colorcode = (255-255*i/(obj_num/3), 0, 0)
        
        if i<=obj_num/3:
            i = int(i-(obj_num/3)+1)
            colorcode = (255-255*i/(obj_num/3), 0,0)
        elif i>obj_num/3 and i<obj_num*2/3:
            i = int(i-(obj_num/3)+1)
            colorcode = (0,255-255*i/(obj_num/3), 0)
        elif i>=obj_num*2/3:
            i = int(i-(obj_num/3)+1)
            colorcode = (0,0,255-255*i/(obj_num/3))
            
        for pos in Obj.Positions:
            cv2.circle(frame, pos[0], 10, colorcode, 5)
            cv2.putText(frame, '%i,%i' % (pos[1],Obj.id), fontFace = cv2.FONT_HERSHEY_SIMPLEX, color = (0,0,0), org = (pos[0][0]+20, pos[0][1]+20), fontScale = 2, thickness = 2)
           
    #add drift direction to image       
    cv2.line(frame, (int(frame_width-200), 200), (int(frame_width-200 - 1000*np.tan(np.radians(MEANDRIFT[0][1]))), 1000+200), (255,0,0), 5)
    save_frame(IMAGE_DIR + "/marked_obj_all.png"
            , frame, "marked all Objs")
    
    
    #take first frame to mark only filtered
    for (i, Obj) in enumerate(filtered_objects):
        obj_num = len(filtered_objects)
        colorcode = (255-255*i/(obj_num/3), 0, 0)
        
        if i<=obj_num/3:
            i = int(i-(obj_num/3)+1)
            colorcode = (255-255*i/(obj_num/3), 0,0)
        elif i>obj_num/3 and i<obj_num*2/3:
            i = int(i-(obj_num/3)+1)
            colorcode = (0,255-255*i/(obj_num/3), 0)
        elif i>=obj_num*2/3:
            i = int(i-(obj_num/3)+1)
            colorcode = (0,0,255-255*i/(obj_num/3))
            
        for pos in Obj.Positions:
            cv2.circle(frame2, pos[0], 10, colorcode, 5)
            cv2.putText(frame2, '%i,%i' % (pos[1],Obj.id), fontFace = cv2.FONT_HERSHEY_SIMPLEX, color = (0,0,0), org = (pos[0][0]+20, pos[0][1]+20), fontScale = 2, thickness = 2)
           
    #add drift direction to image       
    cv2.line(frame2, (int(frame_width-200), 200), (int(frame_width-200 - 1000*np.tan(np.radians(MEANDRIFT[0][1]))), 1000+200), (255,0,0), 5)
    save_frame(IMAGE_DIR + "/marked_obj_filtered.png"
            , frame2, "marked all filtered Objs")
            
    
    print('The mean drift of detected floes is ... (veloc., angle)')
    print(MEANDRIFT)
    log.debug("Done.")






#################################################################################

if __name__ == "__main__":
    """
    statistical tracker: automatically detects ice floe contours, saves them as 
    objects and inits a tracker on it in every next step it assigns and thus 
    updates known objects or creates new added a drift calculation for each single 
    and whole result using for evaluation and result refinement works good so far. 
    Some refinement could be made Thresholds calibrated with a video and not the correct footage
    """
    
    
    log = init_logging()

    if not os.path.exists(IMAGE_DIR):
        log.debug("Creating image directory `%s`...", IMAGE_DIR)
        os.makedirs(IMAGE_DIR)

    main()
    