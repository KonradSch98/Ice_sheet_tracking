import cv2
import os
import matplotlib.pyplot as plt 
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.util import invert
from datetime import datetime
import numpy as np
import glob
import matplotlib.path as Path

from DrawOnImage import Draw_Application




MODES = ['ManuelLine', 'SemiAutomaticLine', 'ManuelROID']
TRACKINGMODE = MODES[2]





def GetDir():
        """
        Asks for a file/directory and checks its existence.
        Args only relevant for embedded use of the class

        Args:
            Dir (str, optional):  directory of the 'file' or 'folder of the files to convert
                    input only relevant for embedded use
                    Defaults to None. >> asks the user
            Height (float, optional):  (list) of the camera height of the pased photo
                    only relevant for type 'file' in embedded use
                    Defaults to None. >> asks user
                    
        Returns:
            Type (str): whether the given path is 'folder' or 'file
        """
        
        print('Please enter the directory or a path of an image:\n')
        input_path = input()
        
        Type = 'folder'
        while Type != 'file':
            if not os.path.isfile(input_path) and not os.path.isdir(input_path):
                print('ERROR: Something went wrong. Please enter the directory/path again:\n')
                input_path = input()
            elif os.path.isfile(input_path):
                print('File found.')        
                ImgPath = input_path.strip(' ') 
                Filename = ImgPath.split("\\")[-1]
                fname = Filename[:len(Filename)-4]               #photo ID
                Gopro_id = fname.split('_')[0]
                ImgDir = os.path.dirname(ImgPath)               #dir of file
                DirSize = len(os.listdir(ImgDir))
                Type = "file"
        
            elif os.path.isdir(input_path):    
                print('You entered a directory. Pls enter a file')        
                input_path = input()   
                Type = "folder"
                
        return ImgPath, fname, Gopro_id, ImgDir, DirSize




#####################################
#####################################




def filter_mask(img):
    """
    create and refine binary mask of frame

    Args:
        img ([np.ndarray]): frame image

    Returns:
        [np.ndarray]: refined binary mask
    """

    
    grey = rgb2gray(img)
    factor = 1.1
    otsu_thresh = threshold_otsu(grey)*factor
    mask = grey<otsu_thresh
    mask = invert(mask)
    mask.dtype='uint8'
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # remove small ice particles / shrink
    erosion = cv2.erode(mask, kernel, iterations = 4)
    
    #resize ice space
    dilation = cv2.dilate(erosion, kernel, iterations = 3)
    
    
    return dilation




#####################################
#####################################




def FindMatchingContour(bb_contours, ref, size, widthtol, lentol):

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




#####################################
#####################################




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




#####################################
#####################################









#####################################
#####################################




def CheckContour(bb_contours, ref, size, tol, Mode=None):
    """
    calls fct FindMatchingContour
    asks for user input if no matching found until found
    > for new drawing or manual

    Args:
        bb_contours (list(np.ndarray)): all contours in the area of 'size'
        ref (np.ndarray): reference contour to match
        size (np.ndarray): cropped chosen image piece
        tol (int): radius of tolerance area around ref contour
        Mode (str, optional): 'Auto' enables more user choices. Defaults to None.

    Returns:
        bool: ret - if process worked or exited by user
        list(np.ndarray): all contour pieces matching the input
    """


    while True:
        matchCont = FindMatchingContour(bb_contours, ref,  size, tol, .5)
        if len(matchCont) > 0:
            break
        else: 
            print("Could not find a contour that matches your drawing on a\n"\
                "sufficiently long distance.\n"\
                    "Please draw again.")
            ref = Draw_Application(size).start()

            #double check user if area ended up bad nevertheless
            if Mode=='Auto':
                print("In case the area ended up bad, you want to select it manually again? y/n/exit")
                inp = input().casefold()
                #inp = inp.casefold()
                if inp not in ('y','n','yes','no', 'e', 'exit'):
                    while True:
                        print("Please enter again y/n/e.")
                        inp = input().casefold()
                        if inp in ('y','n','yes','no', 'e', 'exit'):
                            break
                if inp == 'y' or inp == 'yes':
                    #switch back to manual input
                    return False, []
                
                elif inp == 'e' or inp == 'exit':
                    exit("Exited due to user input in Fct='CheckContour', Mode='SemiAutomaticLine'.")

            binary = filter_mask(size)
            
            bb_contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    return True, matchCont




#####################################
#####################################




def ProcessFrame(img, Mode=None):

    if Mode==None:
        exit("ERROR: No 'ProcessFrame' Mode was passed.")


    #select roi
    cv2.namedWindow('select frame',  cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('select frame', cv2.WND_PROP_TOPMOST, 1)
    bbox =  cv2.selectROI('select frame',img, False)
    cv2.destroyAllWindows()


    #crop and prepare image
    ImgBox = img.copy()
    size = ImgBox[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    tol = int(min(40,size.shape[0]/20, size.shape[1]/20 ))
    binary = filter_mask(size)
    print("Please draw from l->r or from u->d.")
    
    bb_contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE) 
    


    if Mode == 'Roid':
        centroids = []
        printimg = size.copy()
        for i, c in enumerate(bb_contours):
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centroids.append((cx,cy))
            cv2.drawContours(printimg, bb_contours, i, (0,255,0), 1)
            cv2.circle(printimg, (cx,cy), 2, (255,0,0), -1)
            cv2.putText(printimg, str(i), (cx-10, cy-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1 )
        
        cv2.namedWindow('select contour',  cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('select contour', cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow('select contour', printimg)
        cv2.waitKey(2)
        
        print("\nWhich contour do you want?\n"\
            "Enter int.")
        number = input()
        while True:
            try: 
                number = int(number)
                break
            except:
                print('Please enter the int again.')
                number = input() 
        cv2.destroyAllWindows() 

        centroid = (centroids[number][0] + bbox[0], centroids[number][1]+bbox[1])
        return bb_contours[number], bbox, size, centroid



    elif Mode == 'Line':
        drawing = Draw_Application(binary).start()
        ret, matchCont = CheckContour(bb_contours, drawing, size, tol)
        return matchCont, bbox, size
    else:
        exit("ERROR: unknown 'ProcessFrame' Mode was passed.")




#####################################
#####################################




def SemiAutomaticLine(img, index, Contours):
    """
    Asks user for a Roid in the passed frame 
    and to draw a contour in this Roid
    This approach then tries to identify the previously 
    detected contour in the next frame automatically
    If not working switches to manuel -Roid/Draw/detect

    Args:
        img (np.ndarray): current img to process
        index (int): index of frame - indicates if first or later

    Returns:
        list(np.ndarray): all contour pieces matching the input
        list(int): bounding box of contour piece (x,y,w,h)
        np.ndarray: cropped chosen image piece
        tuple(int): centroid of contour by moments (x,y)
    """



    if index==0:
        #first frame simple manuel input
        matchCont, bbox, size = ProcessFrame(img, Mode='Line')

    else:
        
        binary = filter_mask(img)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        del binary 
        #prepare img with contour reference
        RefImg = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
        cv2.drawContours(RefImg, Contours[index-1], -1, (255,255, 0), 3)
        input_bbox = cv2.boundingRect(Contours[index-1])
        RefImg = RefImg[input_bbox[1]:input_bbox[1]+input_bbox[3], input_bbox[0]: input_bbox[0]+input_bbox[2]]
        RefGray = cv2.cvtColor(RefImg, cv2.COLOR_BGR2GRAY)
        w, h = RefImg.shape[1] , RefImg.shape[0]
        del RefImg
        #prepare new bin image with contours
        NewConImg = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
        cv2.drawContours(NewConImg, contours, -1, (255,255, 0), 3)
        ImageGray = cv2.cvtColor(NewConImg, cv2.COLOR_BGR2GRAY)
        del NewConImg

        #try to identtify some matching area
        try:
            #find matching area from ref in new
            result = cv2.matchTemplate(ImageGray, RefGray,cv2.TM_CCOEFF_NORMED)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
            (startX, startY) = maxLoc
            bbox = (startX, startY, w, h)
            
            #show found area for user revision
            imgg = img.copy()
            cv2.rectangle(imgg, (bbox[0],bbox[1]) , (bbox[0]+bbox[2],bbox[1]+bbox[3]), (255,0,0), 3)
            cv2.namedWindow('frame correct?',  cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('frame correct?', cv2.WND_PROP_TOPMOST, 1)
            cv2.imshow('frame correct?',imgg)
            cv2.waitKey(1)

            print("Is this the correct area/contour? y/n")
            inp = input().casefold()
            #inp = inp.casefold()
            if inp not in ('y','n','yes','no'):
                while True:
                    print("Please enter again y/n.")
                    inp = input().casefold()
                    if inp in ('y','n','yes','no'):
                        break
            cv2.destroyAllWindows() 

            if inp == 'y' or inp == 'yes':
                #try to find contour in found area
                ImgBox = img.copy()
                size = ImgBox[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
                tol = int(min(40,size.shape[0]/20, size.shape[1]/20 ))
                diff=(Bboxes[0][0]-startX, Bboxes[0][1]-startY)              #drift from prev. frame
                PrevShiftedMatch = Contours[index-1]-diff                    #drift corrected references
                PrevShiftedMatch = PrevShiftedMatch.reshape(PrevShiftedMatch.shape[0],1,2)
                bb_contours = FindContoursinBbox(bbox, contours)
                ret, matchCont = CheckContour(bb_contours, PrevShiftedMatch, size, tol, Mode='Auto')
                if ret == False:
                   matchCont, bbox, size = ProcessFrame(img, Mode='Line') 
            else: 
                matchCont, bbox, size = ProcessFrame(img, Mode='Line')

        except:
            matchCont, bbox, size = ProcessFrame(img, Mode='Line')
    

    M = cv2.moments(Contours[index-1])
    cx = int(M['m10']/M['m00'])+bbox[0]
    cy = int(M['m01']/M['m00'])+bbox[1]
    
    return matchCont, bbox, size, (cx,cy)




#####################################
#####################################




if __name__ == '__main__':
    
    Centroids = []
    Contours = []
    Bboxes= []
    Times = []
    ShapeSize = []
    Distance = []
    if TRACKINGMODE == 'ManuelROID':
        Area = []
    
    
    for index in (0,1):
    
    
        path, fname, Gopro_id, fdir, DirSize = GetDir()    
        img = cv2.imread(path)


        if TRACKINGMODE == 'ManuelLine':
            matchCont, bbox, size = ProcessFrame(img, Mode='Line')
            M = cv2.moments(matchCont[0])
            cx = int(M['m10']/M['m00'])+bbox[0]
            cy = int(M['m01']/M['m00'])+bbox[1]
            centroid = (cx, cy)
        elif TRACKINGMODE == 'SemiAutomaticLine':
            matchCont, bbox, size, centroid = SemiAutomaticLine(img, index, Contours)
        elif TRACKINGMODE == 'ManuelROID':
            matchCont, bbox, size, centroid = ProcessFrame(img, Mode='Roid')    
        
       
        Centroids.append(centroid)
        Contours.append(matchCont[0]+[bbox[0],bbox[1]]) 
        Bboxes.append(bbox) 
        if TRACKINGMODE == 'ManuelROID':
            Area.append(cv2.contourArea(Contours[index]))


        printimg = img.copy()
        #save processed image
        cv2.drawContours(printimg, Contours[index], -1, (0,255,0), 2)
        #cv2.rectangle(printimg, (bbox[0], bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (255,0,0), 3)
        cv2.circle(printimg, (cx,cy), 2, (255,0,0), -1)
        cv2.putText(printimg, str(index), (cx-10, cy-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1 )
        cv2.imwrite(fdir + "/" + fname + "_cut.jpg" ,printimg)



        #find time stamp
        try:
            time_file = open(os.path.join(fdir, "timedata.txt"), 'r')
            i = 0
            while True:
                next_line = time_file.readline()
                if Gopro_id in next_line:
                    break
                elif i>DirSize:
                    print("ERROR: Could not find timeStamp in timedata.txt file")
                    exit() 
                i=i+1
            time = next_line.split('\t')
            time = time[1].strip('\n')
            datetime_object = datetime.strptime(time, '%Y:%m:%d %H:%M:%S')
            
        except FileNotFoundError:
            print("Please enter the timestamp of the given Image ('%Y:%m:%d %H:%M:%S'):")
            time = input()
            while True:
                try: 
                    datetime_object = datetime.strptime(time, '%Y:%m:%d %H:%M:%S')
                    break
                except:
                    print("Please enter the timestamp again in format '%Y:%m:%d %H:%M:%S'")
                    time = input()
        
        
        

        #find pixel size            
        try:
            size_file = glob.glob(os.path.join(fdir, Gopro_id) + "*.txt")[0]
            size_file = open(size_file, 'r')
            PixelSqSize = float(size_file.readline())
            PixelSize = float(size_file.readline())
            
        except FileNotFoundError:
            print("Please enter the pixelsize in mtr of the given Image:")
            size = input()
            while True:
                try: 
                    PixelSize = float(size)
                    break
                except:
                    print("Please enter the pixelsize in mtr again.")
                    size = input()


        Times.append(datetime_object)
        ShapeSize.append(PixelSqSize*Area[index])
  

   
    deltatime = Times[1]-Times[0]
    print("Time difference: " + str(deltatime.total_seconds()) + " sec")
    dx = Centroids[1][0]-Centroids[0][0]
    dy = Centroids[1][1]-Centroids[0][1]
    distance = np.sqrt(dx**2 + dy**2)
    print("Pixel distance: "+ str(PixelSize*distance) + " m")
    if TRACKINGMODE == 'ManuelROID':
        print("area :" + str(ShapeSize[0]) + " qm  und " + str(ShapeSize[1])+ " qm")
    
    

    print('Done')
    
