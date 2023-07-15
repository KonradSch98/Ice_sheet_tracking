import logging
import logging.handlers
import os
import sys

import cv2
import numpy as np

from cv2_icetrackclass_csrt_reset_drift_manuel import VehicleCounter
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




def FindContoursinBbox(input_bbox, contours):
    
    
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




def FindMatchingContour(bb_contours, ref,frame, widthtol, lentol):

    trsh = np.array([[-widthtol,-widthtol], [widthtol,-widthtol],[widthtol,widthtol], [-widthtol,widthtol]])
    inside_pts = []
    for i, con in enumerate(bb_contours):
        bounding_area = ref-trsh
        coord = ref[:][:][:][:,0]
        x = coord[:,0]
        y = coord[:,1]
        horizontal = (max(x)-min(x))>(max(y)-min(y))
        if horizontal:
            bounding = np.concatenate(([bounding_area[:,0][-1]], np.flipud(bounding_area[:,1]),[bounding_area[:,2][1]], bounding_area[:,3]))
        else:
            bounding = np.concatenate((bounding_area[:,0], [bounding_area[:,1][-1]] , np.flipud(bounding_area[:,2]), [bounding_area[:,3][0]]))

        path = Path.Path(bounding)
        inside_bin = path.contains_points(con)
        inside_pts.append(con[inside_bin])
        
    matchCont = [con for con in inside_pts if len(con)>=lentol*len(ref)]



    
    return matchCont




####################        
#################### 





def detect_vehicles(binary_mask, frame_number, frame):
    
    log = logging.getLogger("detect_objects")

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

        return matchCont[0], (cx,cy), (input_bbox[0], input_bbox[1], input_bbox[0]+input_bbox[2], input_bbox[1]+input_bbox[3])
    
    
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

        #find matching area from ref in new
        result = cv2.matchTemplate(ImageGray, RefGray,cv2.TM_CCOEFF_NORMED)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
        #if maxLoc < .3:
        (startX, startY) = maxLoc
        endX = startX + RefImg.shape[1]
        endY = startY + RefImg.shape[0]
        diff=(input_bbox[0]-startX, input_bbox[1]-startY)       #drift
        
        PrevShiftedMatch = matchCont[0]-diff                    #drift correction
        PrevShiftedMatch = PrevShiftedMatch.reshape(PrevShiftedMatch.shape[0],1,2)
        bb_contours = FindContoursinBbox((startX, startY, endX-startX, endY-startY), contours)
        matchCont = FindMatchingContour(bb_contours, PrevShiftedMatch, frame, tol, .7)
        M = cv2.moments(matchCont[0])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])       
        
        return matchCont[0], (cx,cy), (startX, startY, endX, endY)
        print('hi')


    





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

@staticmethod
def get_vector(a, b):
    """Calculate vector (distance, angle in degrees) from point a to point b.

    Angle ranges from -180 to 180 degrees.
    Vector with angle 0 points straight down on the image.
    Values increase in clockwise direction.
    """
    dx = float(b[0] - a[0])
    dy = float(b[1] - a[1])

    distance = np.sqrt(dx**2 + dy**2)

    if dy > 0:
        angle = np.degrees(np.atan(-dx/dy))
    elif dy == 0:
        if dx < 0:
            angle = 90.0
        elif dx > 0:
            angle = -90.0
        else:
            angle = 0.0
    else:
        if dx < 0:
            angle = 180 - np.degrees(np.atan(dx/dy))
        elif dx > 0:
            angle = -180 - np.degrees(np.atan(dx/dy))
        else:
            angle = 180.0        




    # #korrigiert von mir in mathematisch korrekt
    # if dy > 0:
    #     if dx == 0:
    #         angle = 90
    #     elif dx > 0:
    #         angle = math.degrees(math.atan(dx/dy))
    #     else:
    #         angle = 180 - math.degrees(math.atan(dx/dy))
    # elif dy < 0:
    #     if dx == 0:
    #         angle = -90
    #     elif dx > 0:
    #         angle = -1* math.degrees(math.atan(-dx/dy))
    #     else:
    #         angle = math.degrees(math.atan(dx/dy)) - 180
    # else:
    #     if dx >= 0 :
    #         angle = 0
    #     else:
    #         angle = 180
            
            

    return distance, angle 

    
####################        
#################### 



########################################################################################################################
# make vector calculation for drift 
# angle is missing
# adjust function above
def drift(Objects):
    global MEANDRIFT

    drift_n = MEANDRIFT[0]
    index_n = MEANDRIFT[1]
    
    for i,ob in enumerate(Objects):
        
        contour, centroid, bbox = ob
        ob_p1 = Objects[i+1]
        contour_p1, centroid_p1, bbox_p1 = ob_p1
        dx = float(bbox_p1[0] - bbox[0])
        dy = float(bbox_p1[1] - bbox[1])
        distance = np.sqrt(dx**2 + dy**2)
        
        #####
        vector_n1 = np.array(Vehicle.av_vector[0])
        offset = ( vector_n1 -  drift_n) / (index_n + 1)
        #update drift
        drift_n += offset
        index_n += 1
        
    MEANDRIFT[0] = drift_n
    MEANDRIFT[1] = index_n





####################        
#################### 





def process_frame(frame_number, frame, objects):
    
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
    contour, centroid, bbox = detect_vehicles(fg_mask, frame_number, frame)
    log.debug("Found matching contour")
    
    objects.append([contour, centroid, bbox])

    log.debug("centroid=%s, bounding_box=%s", centroid, bbox)


    # Mark the bounding box and the centroid on the processed frame
    # NB: Fixed the off-by one in the bottom right corner
    cv2.rectangle(processed, (bbox[0], bbox[1]), (bbox[2], bbox[3] - 1), BOUNDING_BOX_COLOUR, 1)
    cv2.circle(processed, centroid, 2, CENTROID_COLOUR, -1)
    cv2.drawContours(processed, [contour], -1, (0, 0, 255), 3)


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






    # Set up image source
    log.debug("Initializing video capture device #%s...", IMAGE_SOURCE)
    cap = cv2.VideoCapture(IMAGE_SOURCE)

    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    log.debug("Video capture frame size=(w=%d, h=%d)", frame_width, frame_height)

    log.debug("Starting capture loop...")
    frame_number = -1
    objects = []
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
                , frame_number, frame, "source frame #%d")


        log.debug("Processing frame #%d...", frame_number)
        processed = process_frame(frame_number, frame, objects)

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
    
    

    # drift(VEHICLES)
    
    # filtered_vehicles = []
    # for veh in VEHICLES:
    #     vector = (np.array(veh.positions[-1][0])-np.array(veh.positions[0][0]))/len(veh.positions)
    #     vector = (abs(vector[0]), vector[1])
    #     if (vector[1]<veh.av_vector[0][1]*1.4 and vector[1]>veh.av_vector[0][1]*.6) or (vector[0]<veh.av_vector[0][0]*1.4 and vector[0]>veh.av_vector[0][0]*.6):
    #         filtered_vehicles.append(veh)
            
    # l=[]
    # for veh in filtered_vehicles:
    #     if (veh.av_vector[0][1]<MEANDRIFT[0][1]*1.5 and veh.av_vector[0][1]>MEANDRIFT[0][1]*.5) or (veh.av_vector[0][0]<MEANDRIFT[0][0]*1.5 and veh.av_vector[0][0]>MEANDRIFT[0][0]*.5):
    #         l.append(veh)
    
    # filtered_vehicles = l
    # del l
    
    # cap = cv2.VideoCapture(IMAGE_SOURCE)
    # ret, frame = cap.read()
    # frame2 = frame.copy()
    
    # for (i, veh) in enumerate(VEHICLES):
    #     veh_num = len(VEHICLES)
    #     color=255-255*i/veh_num
    #     colorcode = (255-255*i/(veh_num/3), 0, 0)
        
    #     if i>veh_num/3 and i<veh_num*2/3:
    #         i = int(i-(veh_num/3)+1)
    #         colorcode = (0,255-255*i/(veh_num/3), 0)
    #     elif i>=veh_num*2/3:
    #         i = int(i-(veh_num/3)+1)
    #         colorcode = (0,0,255-255*i/(veh_num/3))
            
    #     for pos in veh.positions:
    #         cv2.circle(frame, pos[0], 10, colorcode, 5)
    #         cv2.putText(frame, '%i,%i' % (pos[1],veh.id), fontFace = cv2.FONT_HERSHEY_SIMPLEX, color = (0,0,0), org = (pos[0][0]+20, pos[0][1]+20), fontScale = 2, thickness = 2)
           
    # #add drift direction to image       
    # cv2.line(frame, (int(frame_width-200), 200), (int(frame_width-200 - 1000*np.tan(np.radians(MEANDRIFT[0][1]))), 1000+200), (255,0,0), 5)
    
    # save_frame(IMAGE_DIR + "/marked_veh_all%i.png"
    #         ,1, frame, "marked all cars %i")
    
    
    # for (i, veh) in enumerate(filtered_vehicles):
    #     veh_num = len(filtered_vehicles)
    #     color=255-255*i/veh_num
    #     colorcode = (255-255*i/(veh_num/3), 0, 0)
        
    #     if i>veh_num/3 and i<veh_num*2/3:
    #         i = int(i-(veh_num/3)+1)
    #         colorcode = (0,255-255*i/(veh_num/3), 0)
    #     elif i>=veh_num*2/3:
    #         i = int(i-(veh_num/3)+1)
    #         colorcode = (0,0,255-255*i/(veh_num/3))
            
    #     for pos in veh.positions:
    #         cv2.circle(frame2, pos[0], 10, colorcode, 5)
    #         cv2.putText(frame2, '%i,%i' % (pos[1],veh.id), fontFace = cv2.FONT_HERSHEY_SIMPLEX, color = (0,0,0), org = (pos[0][0]+20, pos[0][1]+20), fontScale = 2, thickness = 2)
           
    # #add drift direction to image       
    # cv2.line(frame2, (int(frame_width-200), 200), (int(frame_width-200 - 1000*np.tan(np.radians(MEANDRIFT[0][1]))), 1000+200), (255,0,0), 5)
    
    # save_frame(IMAGE_DIR + "/marked_veh_filtered%i.png"
    #         ,1, frame2, "marked all cars %i")
            
    
    
    log.debug("Done.")






#################################################################################

if __name__ == "__main__":
    log = init_logging()

    if not os.path.exists(IMAGE_DIR):
        log.debug("Creating image directory `%s`...", IMAGE_DIR)
        os.makedirs(IMAGE_DIR)

    main()
    