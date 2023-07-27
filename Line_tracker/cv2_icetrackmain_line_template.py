import logging
import logging.handlers
import os
import sys
import cv2
import numpy as np

from DrawOnImage import Draw_Application
from skimage.filters import threshold_otsu
import matplotlib.path as Path

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




def FindContoursinBbox(input_bbox, contours):
    """
    looks for contour pieces intersecting with bbox area

    Args:
        input_bbox (list(int)): bbox of area of interest
        contours (list(np.ndarray)): all contours in frame 

    Returns:
        list(np.ndarray): all contours in the area of 'size'
    """
    
    
    #reduce all contours to only range of my drawn contour
    bb_contours = []
    for i, con in enumerate(contours):
        lis = con[:][:][:][:,0]
        px = [i for i,p in enumerate(lis[:,0]) if (p > input_bbox[0]-10) and (p<input_bbox[0]+input_bbox[2]+10)]
        py = [i for i,p in enumerate(lis[px][:,1]) if (p > input_bbox[1]-10) and (p<input_bbox[1]+input_bbox[3]+10)]
        if len(py) != 0 and len(px) != 0:
            bb_contours.append(lis[px][py])
    
    return bb_contours





####################        
#################### 




def FindMatchingContour(bb_contours, ref, frame, widthtol, lentol):
    """
    creates a tolerance area around ref contour
    then checks for every con in bb_contours
    which points are within this tolerance
    only returns point groups whos length is within tolerance

    Args:
        bb_contours (list(np.ndarray)): all contours in the area of 'size'
        ref (np.ndarray): reference contour to match
        size (np.ndarray): cropped chosen image piece
        widthtol (int): radius of tolerance area around ref contour
        lentol (int): tolerande of length differenc between reference and found contour
        
    Returns:
        list(np.ndarray): list of matching contours
    """

    #tolerance rectangle
    trsh = np.array([[-widthtol,-widthtol], [widthtol,-widthtol],[widthtol,widthtol], [-widthtol,widthtol]])
    bounding_area = ref-trsh
    coord = ref[:][:][:][:,0]
    x = coord[:,0]
    y = coord[:,1]
    horizontal = (max(x)-min(x))>(max(y)-min(y))
    #stack points in bounding_area in ongoing bounding line
    if horizontal:
        bounding = np.concatenate(([bounding_area[:,0][-1]], np.flipud(bounding_area[:,1]),[bounding_area[:,2][1]], bounding_area[:,3]))
    else:
        bounding = np.concatenate((bounding_area[:,0], [bounding_area[:,1][-1]] , np.flipud(bounding_area[:,2]), [bounding_area[:,3][0]]))
    path = Path.Path(bounding)
    
    #check points in each contour    
    inside_pts = []    
    for i, con in enumerate(bb_contours):    
        con_pts = con[:][:][:][:,0]
        try:
            inside_bin = path.contains_points(con_pts)
            inside_pts.append(con[inside_bin])
        except:
            pass
    
    #filter matched contours
    matchCon = [con for con in inside_pts if len(con)>0]
    matchCont= []
    for con in matchCon:
        a=(con[0]-con[-1])[0]
        b=(ref[0]-ref[-1])[0]
        if np.sqrt(a[0]**2 + a[1]**2) >= lentol*np.sqrt(b[0]**2 + b[1]**2):
            matchCont.append(con)
    
    return matchCont




####################        
#################### 






def DetectObject(binary_mask, frame_number, frame):
    """
    in first frame ask contour line from user and find
    later take prev. contour info
    to find in new frame
    there reduce search area by drift info

    Args:
        binary_mask ([np.ndarray]): binary image mask
        shape ([list(int)]): img dimensions
        frame_number ([int]): index of current frame
        frame ([np.ndarray]): current frame

    Returns:
        [list(lists)]: list of all foundings (bbox, centroid, frame_nr)
    """



    log = logging.getLogger("DetectObject")

    global drawing
    global matchCont
    global tol
    tol = int(min(30,frame.shape[0]/75, frame.shape[1]/75 ))
    
 
    
    # Find the contours of any object in the image
    contours, hierarchy = cv2.findContours(binary_mask
        , cv2.RETR_EXTERNAL
        , cv2.CHAIN_APPROX_SIMPLE)


    if frame_number == 0:
        
        #get drawing input the first time
        drawing = Draw_Application(frame).start()
        input_bbox = cv2.boundingRect(drawing)
        bb_contours = FindContoursinBbox(input_bbox, contours)
        while True:
            if len(bb_contours) > 0:
                break
            else: 
                print("Could not find a contour in the given bbox.\n"\
                        "Please draw again.")
                drawing = Draw_Application(frame).start()
                input_bbox = cv2.boundingRect(drawing)
                bb_contours = FindContoursinBbox(input_bbox, contours)


        #find match to drawing
        while True:
            matchCont = FindMatchingContour(bb_contours, drawing, frame, tol, .5)
            M = cv2.moments(matchCont[0])
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            if len(matchCont) > 0:
                break
            else: 
                print("Could not find a contour that matches your drawing on a\n"\
                    "sufficiently long distance.\n"\
                        "Please draw again.")
                drawing = Draw_Application(frame).start()
                input_bbox = cv2.boundingRect(drawing)
                bb_contours = FindContoursinBbox(input_bbox, contours)

        
        return matchCont[0], (cx,cy), (input_bbox[0], input_bbox[1], input_bbox[2], input_bbox[3])
    
    
    #keep tracking 
    if frame_number > 0:

        #prepare reference and new image information
        RefImg = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
        cv2.drawContours(RefImg, matchCont, -1, (255,255, 0), 3)
        input_bbox = cv2.boundingRect(matchCont[0])
        RefImg = RefImg[input_bbox[1]:input_bbox[1]+input_bbox[3], input_bbox[0]: input_bbox[0]+input_bbox[2]]
        RefGray = cv2.cvtColor(RefImg, cv2.COLOR_BGR2GRAY)
        
        NewConImg = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
        cv2.drawContours(NewConImg, contours, -1, (255,255, 0), 3)
        ImageGray = cv2.cvtColor(NewConImg, cv2.COLOR_BGR2GRAY)


        if frame_number == 1:
            #reduce area by vector information
            contour, centroid, bbox, vector = OBJECTS[-1]
            x,y,w,h = bbox
            area_contour = [centroid[0]-2*w, centroid[1]-2*h, centroid[0]+2*w, centroid[1]+2*h]  #[x1,y1, x2, y2]
            #make area bound by image
            for i, pt in enumerate(area_contour):
                if pt < 0: area_contour[i] = 0
                elif i==2 and pt > ImageGray.shape[1]: area_contour[i] = ImageGray.shape[1]
                elif i==3 and pt > ImageGray.shape[0]: area_contour[i] = ImageGray.shape[0]
            
            ImageGray = ImageGray[area_contour[1]:area_contour[3],area_contour[0]:area_contour[2]]
            result = cv2.matchTemplate(ImageGray, RefGray,cv2.TM_CCOEFF_NORMED)
        else: 
            #reduce area by vector information
            contour, centroid, bbox, vector = OBJECTS[-1]
            reduce_centroid = DoVector(vector, centroid)
            x,y,w,h = bbox
            area_contour = [reduce_centroid[0]-2*w, reduce_centroid[1]-2*h, reduce_centroid[0]+2*w, reduce_centroid[1]+2*h]
            #make area bound by image
            for i, pt in enumerate(area_contour):
                if pt < 0: area_contour[i] = 0
                elif i==2 and pt > ImageGray.shape[1]: area_contour[i] = ImageGray.shape[1]
                elif i==3 and pt > ImageGray.shape[0]: area_contour[i] = ImageGray.shape[0]
            #make area bound by previous position
            for i, pt in enumerate(area_contour):
                if vector[1] > 0 and vector[1] <=90:
                    area_contour[2] = bbox[0]+bbox[2]      #x2 = prev x2
                    area_contour[1] = bbox[1]              #y1 = prev y1
                elif vector[1] > 90:
                    area_contour[2] = bbox[0]+bbox[2]      #x2 = prev x2
                    area_contour[3] = bbox[1]+bbox[3]      #y2 = prev y2
                elif vector[1] < 0 and vector[1] >= -90:
                    area_contour[0] = bbox[0]              #x1 = prev x1
                    area_contour[1] = bbox[1]              #y1 = prev y1                    
                elif vector[1] <-90:
                    area_contour[0] = bbox[0]              #x1 = prev x1
                    area_contour[3] = bbox[1]+bbox[3]      #y2 = prev y2                   
                    
                    
            ImageGray = ImageGray[area_contour[1]:area_contour[3],area_contour[0]:area_contour[2]]
            result = cv2.matchTemplate(ImageGray, RefGray,cv2.TM_CCOEFF_NORMED)

        
        
        
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
        #if maxLoc < .3:
        (startX, startY) = maxLoc
        startX = startX + area_contour[0]
        startY = startY + area_contour[1]
        endX = startX + RefImg.shape[1]
        endY = startY + RefImg.shape[0]
        diff=(input_bbox[0]-startX, input_bbox[1]-startY)       #drift
        
        PrevShiftedMatch = matchCont[0]-diff                    #drift correction
        PrevShiftedMatch = PrevShiftedMatch.reshape(PrevShiftedMatch.shape[0],1,2)
        bb_contours = FindContoursinBbox((startX, startY, RefImg.shape[1], RefImg.shape[0]), contours)
        matchCont = FindMatchingContour(bb_contours, PrevShiftedMatch, frame, tol, .7)
        M = cv2.moments(matchCont[0])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])       
        
        return matchCont[0], (cx,cy), (startX, startY, RefImg.shape[1], RefImg.shape[0])



    


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





def GetVector(pt1, pt2):
    """Calculate vector (distance, angle in degrees) from point pt1 to point pt2.

    Angle ranges from -180 to 180 degrees.
    Vector with angle 0 points straight down on the image.
    Values increase in clockwise direction.

    Args:
        pt1 ([list(int)]): 
        pt2 ([list(int)]): 
    """


    dx = float(pt2[0] - pt1[0])
    dy = float(pt2[1] - pt1[1])

    distance = np.sqrt(dx**2 + dy**2)

    if dy > 0:
        angle = np.degrees(np.arctan(-dx/dy))
    elif dy == 0:
        if dx < 0:
            angle = 90.0
        elif dx > 0:
            angle = -90.0
        else:
            angle = 0.0
    else:
        if dx < 0:
            angle = 180 - np.degrees(np.arctan(dx/dy))
        elif dx > 0:
            angle = -180 - np.degrees(np.arctan(dx/dy))
        else:
            angle = 180.0        



    return distance, angle 

####################        
#################### 





def DoVector(vector, centroid):
    """

    Args:
        pt1 ([list(int)]): 
        pt2 ([list(int)]): 
    """

    distance, angle = vector
    x = -distance * np.sin(np.radians(angle))     
    y = distance * np.cos(np.radians(angle))    
    coord = (x,y)
    new_centroid = [int(a+centroid[i]) for i,a in enumerate(coord)]

    return new_centroid



    
####################        
#################### 



########################################################################################################################
# make vector calculation for drift 
# angle is missing
# adjust function above
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
    
    for i,ob in enumerate(OBJECTS):
        
        contour, centroid, bbox = ob
        ob_p1 = OBJECTS[i+1]
        contour_p1, centroid_p1, bbox_p1 = ob_p1
        dx = float(bbox_p1[0] - bbox[0])
        dy = float(bbox_p1[1] - bbox[1])
        distance = np.sqrt(dx**2 + dy**2)
        
        #####
        vector_n1 = np.array(ob.av_vector[0])
        offset = ( vector_n1 -  drift_n) / (index_n + 1)
        #update drift
        drift_n += offset
        index_n += 1
        
    MEANDRIFT[0] = drift_n
    MEANDRIFT[1] = index_n





####################        
#################### 





def ProcessFrame(frame_number, frame):
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


    #find object in frame
    contour, centroid, bbox = DetectObject(binary_mask, frame_number, frame)
    log.debug("Found matching contour")
    
    
    
    if frame_number == 0:
        vector = 0
    else:
        vector = GetVector(OBJECTS[-1][1], centroid)
    
    
    
    OBJECTS.append([contour, centroid, bbox , vector])

    log.debug("centroid=%s, bounding_box=%s", centroid, bbox)


    # Mark the bounding box and the centroid on the processed frame
    cv2.rectangle(processed, (bbox[0], bbox[1]), (bbox[2], bbox[3] - 1), BOUNDING_BOX_COLOUR, 1)
    cv2.circle(processed, centroid, 2, CENTROID_COLOUR, -1)
    cv2.drawContours(processed, [contour], -1, (0, 0, 255), 3)


    return processed




####################        
#################### 




def main():
    """
    loops over all frames, filteres the recognized Objects
    and saves them all on two viduals
    """


    log = logging.getLogger("main")


    ## könnte hier einen background substractor bauen indem ich von einem anfangsbild die konturen (vom schiff)
    ## speichere und die nächsten male abziehe
    #log.debug("Creating background subtractor...")
    #bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    #log.debug("Pre-training the background subtractor...")
    #default_bg = cv2.imread(IMAGE_FILENAME_FORMAT % 33)
    #bg_subtractor.apply(default_bg, None, 1.0)






    # Set up image source
    log.debug("Initializing video capture device #%s...", IMAGE_SOURCE)
    cap = cv2.VideoCapture(IMAGE_SOURCE)

    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    log.debug("Video capture frame size=(w=%d, h=%d)", frame_width, frame_height)



    log.debug("Starting capture loop...")
    frame_number = -1
    global OBJECTS
    OBJECTS = []
    while True:
        frame_number += 1
        log.debug("Capturing frame #%d...", frame_number)
        ret, frame = cap.read()
        if not ret:
            log.error("Frame capture failed, stopping...")
            break

        log.debug("Got frame #%d: shape=%s", frame_number, frame.shape)

        # Archive raw frames from video to disk for later inspection/testing
        if CAPTURE_FROM_VIDEO:
            save_frame(IMAGE_FILENAME_FORMAT
                , frame, "source frame #%d", frame_number)

        #find and treat all Objects 
        log.debug("Processing frame #%d...", frame_number)
        processed = ProcessFrame(frame_number, frame)

        #archive frame with all valid foundings
        save_frame(IMAGE_DIR + "/processed_%04d.png"
            , processed, "processed frame #%d", frame_number)

    
    log.debug("Done.")






#################################################################################

if __name__ == "__main__":
    """
    Line tracker: kind of manuel statistical tracker useful if no little seperated ice floes are 
    visible can thus track huge floes drifting through the FOV the user draws a line on first frame 
    which should cover a contour segmet of some ice floe the software detects and assigns the correct 
    in-image contour to the drawn line on future frames detects a similar contour segment and so tracks this drawn line
    
    This approach uses a template matching package --> really instable (trans,rot, scale)
    """
    
    
    log = init_logging()

    if not os.path.exists(IMAGE_DIR):
        log.debug("Creating image directory `%s`...", IMAGE_DIR)
        os.makedirs(IMAGE_DIR)

    main()
    