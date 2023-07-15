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











def GetDir():

        """
        Asks for a directory and checks its existence.
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




def filter_mask(img):
    
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




def FindMatchingContour(bb_contours, ref, binary, size, widthtol, lentol):

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
        ######################################################################################
        try:
            inside_bin = path.contains_points(con)
            inside_pts.append(con[inside_bin])
        except:
            pass
        
    matchCont = [con for con in inside_pts if len(con)>=lentol*len(ref)]
    
    return matchCont





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





if __name__ == '__main__':
    
    centroid = []
    contour = []
    times = []
    ShapeSize = []
    Distance = []
    
    
    for index in (0,1):
        path, fname, Gopro_id, fdir, DirSize = GetDir()
        
        #select ROI
        img = cv2.imread(path)


        if index >0:
            
            binary = filter_mask(img)
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE) 
            #prepare reference and new image information
            RefImg = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
            cv2.drawContours(RefImg, matchCont, -1, (255,255, 0), 3)
            input_bbox = cv2.boundingRect(matchCont[0])
            RefImg = RefImg[input_bbox[1]:input_bbox[1]+input_bbox[3], input_bbox[0]: input_bbox[0]+input_bbox[2]]
            RefGray = cv2.cvtColor(RefImg, cv2.COLOR_BGR2GRAY)
            
            NewConImg = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
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
            matchCont = FindMatchingContour(bb_contours, drawing, tol, .7)
            M = cv2.moments(matchCont[0])
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])   
                
            
 
        cv2.namedWindow('select frame',  cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('select frame', cv2.WND_PROP_TOPMOST, 1)
        bbox =  cv2.selectROI('select frame',img, False)
        cv2.destroyAllWindows() 
        
        #crop and prepare image
        ImgBox = img.copy()
        size = ImgBox[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        tol = int(min(40,size.shape[0]/20, size.shape[1]/20 ))
        
        print("Please draw from l->r or from u->d.")
        drawing = Draw_Application(size).start()
        binary = filter_mask(size)
        bb_contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE) 
            
            
            
        while True:
            matchCont = FindMatchingContour(bb_contours, drawing, binary, size, tol, .5)
            if len(matchCont) > 0:
                break
            else: 
                print("Could not find a contour that matches your drawing on a\n"\
                    "sufficiently long distance.\n"\
                        "Please draw again.")
                drawing = Draw_Application(size).start()
                binary = filter_mask(size)
                bb_contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

        
        M = cv2.moments(matchCont[0])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        
        centroid.append((cx,cy))
        contour.append(matchCont[0])  
        printimg = size.copy()

        #save processed image
        cv2.drawContours(printimg, contour[index], i, (0,255,0), 2)
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
            print("Please enter the timestamp of the given Image:")
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


        times.append(datetime_object)
  

   
    deltatime = times[1]-times[0]
    print("Time difference: " + str(deltatime.total_seconds()) + " sec")
    dx = centroid[1][0]-centroid[0][0]
    dy = centroid[1][1]-centroid[0][1]
    distance = np.sqrt(dx**2 + dy**2)
    print("Pixel distance: "+ str(PixelSize*distance) + " m")

    

    print('hi')
    
