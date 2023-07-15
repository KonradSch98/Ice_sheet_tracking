import cv2
import os
import matplotlib.pyplot as plt 
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.util import invert
from datetime import datetime
import numpy as np
import glob











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




if __name__ == '__main__':
    
    centroid = []
    contour = []
    area = []
    times = []
    ShapeSize = []
    for index in (0,1):
        path, fname, Gopro_id, fdir, DirSize = GetDir()
        
        
        
        img = cv2.imread(path)
        cv2.namedWindow('select frame',  cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('select frame', cv2.WND_PROP_TOPMOST, 1)
        bbox =  cv2.selectROI('select frame',img, False)
        cv2.destroyAllWindows() 
        ImgBox = img.copy()
        size = ImgBox[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        binary = filter_mask(size)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        printimg = size.copy()
        for i, c in enumerate(contours):

      
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centroids.append((cx,cy))
            cv2.drawContours(printimg, contours, i, (0,255,0), 1)
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
        
        
       
        cen_global = centroids[number]
        cen_global = (bbox[0] + cen_global[0], bbox[1] + cen_global[1])
        contour.append(contours[number])  
        centroid.append(cen_global)
        area.append(cv2.contourArea(contour[index]))
        
        printimg = size.copy()
        cv2.drawContours(printimg, contour[index], i, (0,255,0), 2)
        cv2.circle(printimg, centroid[index], 4, (255,0,0), -1)
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
        ShapeSize.append(PixelSqSize*area[index])
        
    deltatime = times[1]-times[0]
    print("Time difference: " + str(deltatime.total_seconds()) + " sec")
    dx = centroid[1][0]-centroid[0][0]
    dy = centroid[1][1]-centroid[0][1]
    distance = np.sqrt(dx**2 + dy**2)
    print("Pixel distance: "+ str(PixelSize*distance) + " m")
    print("area :" + str(ShapeSize[0]) + " qm  und " + str(ShapeSize[1])+ " qm")
    

    print('hi')
    
