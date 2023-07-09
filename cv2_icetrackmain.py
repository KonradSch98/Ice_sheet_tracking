import logging
import logging.handlers
import os
import sys

import cv2
import numpy as np

from cv2_icetrackclass_csrt_reset_drift import VehicleCounter
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
VEHICLES = None

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




def save_frame(file_name_format, frame_number, frame, label_format):
    
    file_name = file_name_format % frame_number
    label = label_format % frame_number

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




def detect_vehicles(fg_mask,shape, frame_number, frame):
    
    log = logging.getLogger("detect_vehicles")

    MIN_CONTOUR_WIDTH = 80
    MIN_CONTOUR_HEIGHT = 80
    MAX_CONTOUR_WIDTH = shape[1]/2
    MAX_CONTOUR_HEIGHT = shape[0]/2
    
    
    # Find the contours of any vehicles in the image
    contours, hierarchy = cv2.findContours(fg_mask
        , cv2.RETR_EXTERNAL
        , cv2.CHAIN_APPROX_SIMPLE)

    log.debug("Found %d vehicle contours.", len(contours))

    matches = []
    for (i, contour) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        contour_valid = (w >= MIN_CONTOUR_WIDTH) and (h >= MIN_CONTOUR_HEIGHT) and (w <= MAX_CONTOUR_WIDTH) and (h <= MAX_CONTOUR_HEIGHT)

        if contour_valid == True:
            log.debug("Contour #%d: pos=(x=%d, y=%d) size=(w=%d, h=%d) valid=%s"
                , i, x, y, w, h, contour_valid)

        if not contour_valid:
            continue

        centroid = get_centroid(x, y, w, h)

        matches.append(((x, y, w, h), centroid, frame_number))

    return matches





####################        
#################### 




def filter_mask(fg_mask):
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # remove small ice particles / shrink
    erosion = cv2.erode(fg_mask, kernel, iterations = 4)
    
    #resize ice space
    dilation = cv2.dilate(erosion, kernel, iterations = 2)
    
    
    return dilation




####################        
#################### 




def drift(Vehicles):
    global MEANDRIFT

    drift_n = MEANDRIFT[0]
    index_n = MEANDRIFT[1]
    
    for Vehicle in Vehicles:

        vector_n1 = np.array(Vehicle.av_vector[0])
        offset = ( vector_n1 -  drift_n) / (index_n + 1)
        #update drift
        drift_n += offset
        index_n += 1
        
    MEANDRIFT[0] = drift_n
    MEANDRIFT[1] = index_n





####################        
#################### 





def process_frame(frame_number, frame, car_counter):
    
    global VEHICLES 
    
    
    log = logging.getLogger("process_frame")

    # Create a copy of source frame to draw into
    processed = frame.copy()

    # Remove the background
    #fg_mask = bg_subtractor.apply(frame, None, 0.01)
    
    grey = rgb2gray(processed)
    otsu_thresh = threshold_otsu(grey)*1.1
    fg_mask = grey<otsu_thresh
    
    #binary version
    fg_mask = invert(fg_mask)
    fg_mask.dtype='uint8'
    fg_mask = filter_mask(fg_mask)
    
    #contur version
    #fg_mask = fg_mask ^ ndimage.binary_dilation(fg_mask)
    
    save_frame(IMAGE_DIR + "/mask_%04d.png"
        , frame_number, fg_mask*255, "foreground mask for frame #%d")
    
    #fg_mask.dtype='uint8'
    #fg_mask = filter_mask(fg_mask)

    #fg_mask = invert(fg_mask)
    matches = detect_vehicles(fg_mask, frame.shape, frame_number, frame)


    log.debug("Updating vehicle count...")
    VEHICLES = car_counter.update_count(matches, frame_number, fg_mask, processed)



    matches = []
    for veh in VEHICLES:
        if veh.last_position[1] == frame_number:
            matches.append((veh.last_contour, veh.last_position[0], veh.last_position[1]))
    
    



    log.debug("Found %d valid vehicle contours.", len(matches))
    for (i, match) in enumerate(matches):
        contour, centroid, frame_number = match

        log.debug("Valid vehicle contour #%d: centroid=%s, bounding_box=%s", i, centroid, contour)

        x, y, w, h = contour

        # Mark the bounding box and the centroid on the processed frame
        # NB: Fixed the off-by one in the bottom right corner
        cv2.rectangle(processed, (x, y), (x + w - 1, y + h - 1), BOUNDING_BOX_COLOUR, 1)
        cv2.circle(processed, centroid, 2, CENTROID_COLOUR, -1)
        cv2.putText(processed, str(frame_number), fontFace = cv2.FONT_HERSHEY_SIMPLEX, color = (0,0,0), org = (centroid[0]+20, centroid[1]+20), fontScale = 3, thickness = 2)
    

    return processed




####################        
#################### 




def main():
    log = logging.getLogger("main")


    ## könnte hier einen background substractor bauen indem ich von einem anfangsbild die konturen (vom schiff)
    ##speichere und die nächsten male abziehe
    log.debug("Creating background subtractor...")
    #bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    log.debug("Pre-training the background subtractor...")
    #default_bg = cv2.imread(IMAGE_FILENAME_FORMAT % 33)
    #bg_subtractor.apply(default_bg, None, 1.0)





    car_counter = None # Will be created after first frame is captured

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

        if car_counter is None:
            # We do this here, so that we can initialize with actual frame size
            log.debug("Creating vehicle counter...")
            car_counter = VehicleCounter(frame.shape[:2], frame.shape[0] / 7)


        # Archive raw frames from video to disk for later inspection/testing
        if CAPTURE_FROM_VIDEO:
            save_frame(IMAGE_FILENAME_FORMAT
                , frame_number, frame, "source frame #%d")


        log.debug("Processing frame #%d...", frame_number)
        processed = process_frame(frame_number, frame, car_counter)



        save_frame(IMAGE_DIR + "/processed_%04d.png"
            , frame_number, processed, "processed frame #%d")



        cv2.namedWindow('Source Image',  cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow('Processed Image',  cv2.WINDOW_KEEPRATIO)
        cv2.imshow('Source Image', frame)
        cv2.imshow('Processed Image', processed)

        log.debug("Frame #%d processed.", frame_number)

        c = cv2.waitKey(WAIT_TIME)
        if c == 27:
            log.debug("ESC detected, stopping...")
            break

    log.debug("Closing video capture device...")
    cap.release()
    cv2.destroyAllWindows()
    
    
    VEHICLES[:] = [ v for v in VEHICLES
            if len(v.positions)>=4]
    
    drift(VEHICLES)
    
    filtered_vehicles = []
    for veh in VEHICLES:
        vector = (np.array(veh.positions[-1][0])-np.array(veh.positions[0][0]))/len(veh.positions)
        vector = (abs(vector[0]), vector[1])
        if (vector[1]<veh.av_vector[0][1]*1.4 and vector[1]>veh.av_vector[0][1]*.6) or (vector[0]<veh.av_vector[0][0]*1.4 and vector[0]>veh.av_vector[0][0]*.6):
            filtered_vehicles.append(veh)
            
    l=[]
    for veh in filtered_vehicles:
        if (veh.av_vector[0][1]<MEANDRIFT[0][1]*1.5 and veh.av_vector[0][1]>MEANDRIFT[0][1]*.5) or (veh.av_vector[0][0]<MEANDRIFT[0][0]*1.5 and veh.av_vector[0][0]>MEANDRIFT[0][0]*.5):
            l.append(veh)
    
    filtered_vehicles = l
    del l
    
    cap = cv2.VideoCapture(IMAGE_SOURCE)
    ret, frame = cap.read()
    frame2 = frame.copy()
    
    for (i, veh) in enumerate(VEHICLES):
        veh_num = len(VEHICLES)
        color=255-255*i/veh_num
        colorcode = (255-255*i/(veh_num/3), 0, 0)
        
        if i>veh_num/3 and i<veh_num*2/3:
            i = int(i-(veh_num/3)+1)
            colorcode = (0,255-255*i/(veh_num/3), 0)
        elif i>=veh_num*2/3:
            i = int(i-(veh_num/3)+1)
            colorcode = (0,0,255-255*i/(veh_num/3))
            
        for pos in veh.positions:
            cv2.circle(frame, pos[0], 10, colorcode, 5)
            cv2.putText(frame, '%i,%i' % (pos[1],veh.id), fontFace = cv2.FONT_HERSHEY_SIMPLEX, color = (0,0,0), org = (pos[0][0]+20, pos[0][1]+20), fontScale = 2, thickness = 2)
           
    #add drift direction to image       
    cv2.line(frame, (int(frame_width-200), 200), (int(frame_width-200 - 1000*np.tan(np.radians(MEANDRIFT[0][1]))), 1000+200), (255,0,0), 5)
    
    save_frame(IMAGE_DIR + "/marked_veh_all%i.png"
            ,1, frame, "marked all cars %i")
    
    
    for (i, veh) in enumerate(filtered_vehicles):
        veh_num = len(filtered_vehicles)
        color=255-255*i/veh_num
        colorcode = (255-255*i/(veh_num/3), 0, 0)
        
        if i>veh_num/3 and i<veh_num*2/3:
            i = int(i-(veh_num/3)+1)
            colorcode = (0,255-255*i/(veh_num/3), 0)
        elif i>=veh_num*2/3:
            i = int(i-(veh_num/3)+1)
            colorcode = (0,0,255-255*i/(veh_num/3))
            
        for pos in veh.positions:
            cv2.circle(frame2, pos[0], 10, colorcode, 5)
            cv2.putText(frame2, '%i,%i' % (pos[1],veh.id), fontFace = cv2.FONT_HERSHEY_SIMPLEX, color = (0,0,0), org = (pos[0][0]+20, pos[0][1]+20), fontScale = 2, thickness = 2)
           
    #add drift direction to image       
    cv2.line(frame2, (int(frame_width-200), 200), (int(frame_width-200 - 1000*np.tan(np.radians(MEANDRIFT[0][1]))), 1000+200), (255,0,0), 5)
    
    save_frame(IMAGE_DIR + "/marked_veh_filtered%i.png"
            ,1, frame2, "marked all cars %i")
            
    
    
    log.debug("Done.")






#################################################################################

if __name__ == "__main__":
    log = init_logging()

    if not os.path.exists(IMAGE_DIR):
        log.debug("Creating image directory `%s`...", IMAGE_DIR)
        os.makedirs(IMAGE_DIR)

    main()
    