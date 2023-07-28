import math
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging
matplotlib.use('TKAgg')




#################################################################################




class MultiTracker(object):
    
    def __init__(self, id, position, contour, frame) -> None:
        """
        Class initializing and updating a tracker 

        Args:
            id ([int]): id of tracker
            position ([list(int)]): centroid of ROID
            contour ([list(int)]): Roid of Object to track
            frame ([np.dnarray]): frame on which tot track

        Returns:
            [None]: if tracker init failed
        """


        self.id = id
        
        self.Tracker = cv2.legacy.TrackerCSRT_create()
        self.remove = False
        ret = self.Tracker.init(frame, contour)
        if not ret:
            print('cannot init tracker of id %i' % self.id)
            return None
        self.position = [0,0]
        self.position[1] = position
        self.Contour = [0,0]        
        self.Contour[1] = contour
        self.UpdateCond = True
        


    def Update(self, frame):
        """
        update tracker of id on new frame
        store its centroid and contour
        always maintaining info of one frame back

        Args:
            frame ([np.ndarray]): new frame to track on

        Returns:
            [bool]: if update succeeded
            [list(int)]: contour of new bbox
            [list(int)]: coord of new centroid
            [int]: id of updated tracker
        """

        ret, bbox = self.Tracker.update(frame)

        bbox = [int(i) for i in bbox]
        (x, y, w, h) = bbox
        x1 = int(w / 2)
        y1 = int(h / 2)
        cx = x + x1
        cy = y + y1
        centroid =  (int(cx), int(cy))
        self.position[0] = self.position[1]
        self.position[1] = centroid
        self.Contour[0] = self.Contour[1]
        self.Contour[1] = bbox
        
        return ret, bbox, centroid, self.id




#################################################################################
#################################################################################




class ObjectClass(object):
    def __init__(self, id, position, last_frame, contour):
        """
        Class defining each individual Object that is tracked
        offers different methods to read or write properties

        Args:
            id ([int]): ID of this new Object
            position ([list(int)]): its current centroids coord
            last_frame ([int): frame number of init
            contour ([type]): its current bbox contour
        """

        self.id = id
        self.Positions = [(position, last_frame)]
        self.FramesSinceSeen = 0
        self.counted = False
        self.Constant = False
        self.Contours = [contour]
        self.LastVector = 0
        self.MeanVector = np.zeros(2,list)

        
    @property
    def LastPosition(self):
        """
        returns last position of Object
        Returns:
            [list(int)]: last position of Object
        """
        return self.Positions[-1]

    @property
    def LastContour(self):
        """
        returns last contour of Object
        Returns:
            [list(int)]: last contour of Object
        """
        return self.Contours[-1]

    def AddPosition(self, new_position):
        """
        adds new position to Object
        """
        self.Positions.append(new_position)
        self.FramesSinceSeen = 0
        
    def AddContour(self, new_contour):
        """
        adds new contour to Object
        """
        self.Contours.append(new_contour)

    def DriftUpdate(self):
        """
        updates the average drift direction and length of Object
        """
        drift_n = self.MeanVector[0]
        index_n = self.MeanVector[1]
        vector_n1 = np.array(self.LastVector)
        offset = ( vector_n1 -  drift_n) / (index_n + 1)
        #update drift
        self.MeanVector[0] += offset
        self.MeanVector[1] += 1
        




#################################################################################
#################################################################################




class Const_Object(object):
    def __init__(self, id, position):
        """
        creates an Object that does not move for reference

        Args:
            id ([int]): id of object
            position ([list(int)]): its position
        """

        self.id = id
        self.position = position




#################################################################################
#################################################################################





class ObjectTrackerClass(object):


    def __init__(self, shape):
        """
        init class for global use 
        to pass information structure used
        in this process

        Args:
            shape ([type]): [description]
        """
        self.log = logging.getLogger("ObjectTrackerClass")
        self.height, self.width = shape
        self.Objects = []
        self.Trackers = []
        self.ConstObjects = []
        self.next_vehicle_id = 0
        self.next_const_vehicle_id = 0
        self.vehicle_count = 0





##########################        
##########################    





    @staticmethod
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

        distance = math.sqrt(dx**2 + dy**2)

        if dy > 0:
            angle = math.degrees(math.atan(-dx/dy))
        elif dy == 0:
            if dx < 0:
                angle = 90.0
            elif dx > 0:
                angle = -90.0
            else:
                angle = 0.0
        else:
            if dx < 0:
                angle = 180 - math.degrees(math.atan(dx/dy))
            elif dx > 0:
                angle = -180 - math.degrees(math.atan(dx/dy))
            else:
                angle = 180.0        


        # #korrigiert von mir in mathematisch korrekt
        # this version starts spinning from pos x-axis in +y direction
        # in current version the thresholds are not adjusted to this angle version
        # also drift does not yet work with this one
        #         # if dy > 0:
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





##########################        
##########################    





    @staticmethod
    def IsValidVector(Object,new_vector, contour, tracker=None):
        """
        checks if new vector is similar to old vector

        Args:
            Object ([ObjectClass]): Known Object to compare with
            new_vector ([list(float)]): new vector
            contour ([list(int)]): new contour
            tracker ([Multitracker Class], optional): tracker of Object
                                Defaults to None.

        Returns:
            [bool]: similar or not
        """


        
        old_vector = Object.LastVector
        FramesSinceSeen = Object.FramesSinceSeen
        
        
        if type(old_vector) is tuple:
            old_distance, old_angle = old_vector
            new_distance, new_angle = new_vector
            
            #switch quartents
            if new_angle < 0 and old_angle < 0:
                new_angle = -1*new_angle
                old_angle = -1*old_angle
            if FramesSinceSeen == 0:
                #takes shiftingind distance by different contour sizes into account (2 iterations back)
                #best choice to take only contour width?!?!?!
                upper_d = (old_distance + abs(contour[2]-Object.Contours[-1][2])/3 + abs(Object.Contours[-1][2]-Object.Contours[-2][2])/3 ) *1.3
                lower_d = (old_distance - abs(contour[2]-Object.Contours[-1][2])/3 - abs(Object.Contours[-1][2]-Object.Contours[-2][2])/3 ) *.7                
                # upper_d = 1.4 * old_distance
                # lower_d = .6 * old_distance
                upper_a = 1.2 * old_angle
                lower_a = 0.8 * old_angle
            elif FramesSinceSeen <= 2 :
                #for distant vectors extent angle range by last_seen 
                #is not math correct extetion of the simple but works good until last_seen<=2 (so now third)
                FramesSinceSeen += 1
                upper_d = 1.4**FramesSinceSeen * FramesSinceSeen*old_distance
                lower_d = 0.6**FramesSinceSeen * FramesSinceSeen*old_distance
                upper_a = 1.2**FramesSinceSeen * old_angle
                lower_a = 0.8**FramesSinceSeen * old_angle
            else: return False
            

            #for trackers only check for distance (necessary?)
            if tracker == 'tracker':
                if new_distance <= upper_d and new_distance >= lower_d:
                    return True
            else:     
                if new_distance <= upper_d and new_distance >= lower_d:
                    if new_angle <= upper_a and new_angle >= lower_a:
                        return True
            #make exception for closeness (necessary?)
            if new_distance<=30:
                return True
            else: return False
        
        #mostly relevant for init when vector still 0
        elif new_vector[0]<= 300:
            #initiate by closeness
            return True
        
        else: return False





##########################        
##########################    





    @staticmethod
    def IsValidContour(old_contour, new_contour):
        """
        checks if new contour is similar to old one

        Args:
            old_contour ([np.ndarray]): old contour
            new_contour ([np.ndarray]): new contour

        Returns:
            [bool]: similar or not
        """
    
        if type(old_contour) is tuple:
            x_o, y_o, w_o, h_o = old_contour
            x_n, y_n, w_n, h_n = new_contour
            
            if w_n <= w_o *1.5 and w_n >= w_o *.5:
                if h_n <= h_o *1.5 and h_n >= h_o *.5:
                    return True
                else: return False
            else: return False
            

    

 
 
##########################        
##########################    

   




    def PrintMarker(self,markers, outputimg):
        """
        for debugging
        mark current and last tracker
        mark last Obj position
        mark position of current founding

        Args:
            markers ([list]): centroid and contour of different things
            outputimg ([np.ndarray]): img to mark on

        Returns:
            [np.ndarray]: marked-on image
        """
        

        # Mark the current tracker
        contour, centroid, last_frame = markers[0]
        x, y, w, h = contour
        cv2.rectangle(outputimg, (x-6, y-6), (x + w - 6, y + h - -6), (0,255,0), 2)
        cv2.circle(outputimg, centroid, 3, (0,255,0), -1)
        cv2.putText(outputimg, 'bbox', fontFace = cv2.FONT_HERSHEY_SIMPLEX, color = (0,0,0), org = (centroid[0]+20, centroid[1]+20), fontScale = 3, thickness = 2)
        #ax.imshow(outputimg)
        #cv2.waitKey(0)
        

        # Mark the current founding (or currtracker of none)     
        contour, centroid, last_frame = markers[1]
        x, y, w, h = contour
        cv2.rectangle(outputimg, (x-6, y-6), (x + w - 6, y + h - -6), (255,0,0), 2)
        cv2.circle(outputimg, centroid, 3, (255,0,0), -1)
        cv2.putText(outputimg, 'founding', fontFace = cv2.FONT_HERSHEY_SIMPLEX, color = (0,0,0), org = (centroid[0]+20, centroid[1]+20), fontScale = 3, thickness = 2)
        #ax.imshow(outputimg)
        #cv2.waitKey(0)
        
     
        # Mark the last Object Data
        contour, (centroid, last_frame) = markers[2]
        x, y, w, h = contour
        cv2.rectangle(outputimg, (x, y), (x + w - 1, y + h - 1), (255,0,0), 2)
        cv2.circle(outputimg, centroid, 10, (255,0,0), -1)
        cv2.putText(outputimg, 'l_obj', fontFace = cv2.FONT_HERSHEY_SIMPLEX, color = (0,0,0), org = (centroid[0]+20, centroid[1]+20), fontScale = 3, thickness = 2)
        

        # Mark the last tracker position        
        contour, centroid = markers[3]
        x, y, w, h = contour
        cv2.rectangle(outputimg, (x, y), (x + w - 1, y + h - 1), (0,255,0), 2)
        cv2.circle(outputimg, centroid, 10, (0,255,0), -1)
        cv2.putText(outputimg, 'l_track', fontFace = cv2.FONT_HERSHEY_SIMPLEX, color = (0,255,0), org = (centroid[0]+20, centroid[1]+20), fontScale = 3, thickness = 2)
        
        
        
        return outputimg
 




##########################        
##########################    


   
    
     
    def ResetTracker(self, tracker, id, centroid, contour, frame):
        """
        remove current tracker and make new init
        """
        tracker.remove = True
        reset_tracker = MultiTracker(id, centroid, contour, frame)
        self.Trackers.append(reset_tracker)





##########################        
##########################    

   

    
     
    def CheckTracker(self, id, bbox, centroid, contour, tracker, frame):
        """
        If tracker is lost or inaccurate: remove and make new init

        Args:
            id ([int]): index of Object
            bbox ([type]): tracker of Object in curr. frame,
            centroid ([tuple(float)]): centroid of better match
            contour ([np.ndarray]): contour of beter match
            tracker ([MultiTracker Class]): tracker of Object
            frame ([np.ndarray]): current frame img
        """

        
        #if tracker lost: remove and new init
        if bbox is None:
            self.ResetTracker(tracker, id, centroid, contour, frame)
        
        else:
            bbox_vector = self.GetVector(bbox[1], centroid)
            #if tracker bad, remove and new init
            if bbox_vector[0] > 50:
                self.ResetTracker( tracker, id, centroid, contour, frame)
            else: tracker.UpdateCond = True 
 




##########################        
##########################    

   

    
    
    def UpdateObject(self, Object,foundings, frame=None):
        """
        See if any founding or else tracker matches this Object

        Args:
            Object ([ObjectClass]): current object to check
            foundings ([list(contour, centroid, last_frame)]): list of found objects in frame
            frame ([np.ndarray], optional): current frame img. 
                                    Defaults to None.

        Returns:
            [int] = index of found matching 
                    --- else: NONE

        """


        bbox = None         #corresponding current tracker
        tracker = None      #corresp. tracker
        id = Object.id
        
        #find corresponding bboxes / tracker
        for bbox in self.TrBBoxes:
            if bbox[2] == id:
                break
        for tracker in self.Trackers:
            if tracker.id == id:
                break


    ########
        #search in all foundings for match
        for i, found_ob in enumerate(foundings):
            contour, centroid, last_frame = found_ob
            
            #vector betw Object and founding
            vector = self.GetVector(Object.LastPosition[0], centroid)
            
            
        ######
            #ignore equal pairs
            if vector != 0 and vector !=(0,0):

                val_vec = self.IsValidVector(Object,vector, contour)
                val_con = self.IsValidContour(Object.LastContour, contour)

                #valid match...
                if  val_vec and val_con:

                    ### only for debugging purposes
                    #if bbox is not None:
                        #copy = frame.copy()
                        #frame = self.PrintMarker([bbox, found_ob, (Object.LastContour, Object.LastPosition), (tracker.Contour[0], tracker.position[0])], copy)

                    if Object.FramesSinceSeen > 0:
                        #the distance to a lost Object is (FramesSinceSeen+1) x longer
                        #reduce to mean distance
                        Object.LastVector = (vector[0]/(Object.FramesSinceSeen+1),vector[1])
                    else: Object.LastVector = vector    #update vector
                        
                    #update vehicle
                    Object.AddPosition((centroid, self.CurrFrameNumber))
                    Object.AddContour(contour)
                    
                    self.CheckTracker(id, bbox, centroid, contour, tracker, frame)
                    
                    #remove bbox due to found
                    if bbox is not None:
                        self.TrBBoxes.remove(bbox)  
                                          
                    self.log.debug("Added found_ob (%d, %d) to Object #%d. vector=(%0.2f,%0.2f)"
                        , centroid[0], centroid[1], Object.id, vector[0], vector[1])
                    
                    return i
                

        ######    
            #take not moving object 
            elif vector == 0 or vector ==(0,0):
                val_con = self.IsValidContour(Object.LastContour, contour)

                if val_con:

                    Object.LastVector = vector    
                    Object.AddPosition((centroid, self.CurrFrameNumber))
                    Object.AddContour(contour)
                    
                    #reset tracker for constant object (necessary?)
                    self.ResetTracker(tracker, id, centroid, contour, frame)
                    
                    if bbox is not None:
                        self.TrBBoxes.remove(bbox)  

                    self.log.debug("Added (not moving) found_ob (%d, %d) to Object #%d. vector=(%0.2f,%0.2f)"
                        , centroid[0], centroid[1], Object.id, vector[0], vector[1])
                
                return i



    ########
        #if no detector found, check if tracker matches
        if bbox is not None:

            centroid = bbox[1]
            contour = tuple(bbox[0])

            vector = self.GetVector(Object.LastPosition[0], centroid)
            
            #check if tracker is doing good
            val_vec = self.IsValidVector(Object,vector, contour, 'tracker')
            val_con = self.IsValidContour(Object.LastContour, contour)
            
            if  val_vec and val_con:
                
                #only for debugging
                # copy = frame.copy()
                # frame = self.PrintMarker([bbox, bbox, (Object.LastContour, Object.LastPosition), (tracker.Contour[0], tracker.position[0])], copy)
                # for i, found_ob in enumerate(foundings):
                #     contour, centroid, last_frame = found_ob
                #     x, y, w, h = contour
                #     cv2.rectangle(frame, (x, y), (x + w - 3, y + h - 3), (0,2,255), 2)
                
                centroid = bbox[1]
                contour = tuple(bbox[0])
                
                if Object.FramesSinceSeen > 0:
                    Object.LastVector = (vector[0]/(Object.FramesSinceSeen+1),vector[1])
                else: Object.LastVector = vector 
                
                #update vehicle
                Object.AddPosition((centroid, self.CurrFrameNumber))
                Object.AddContour(contour) 
                
                self.TrBBoxes.remove(bbox)
                tracker.UpdateCond = True
                return None
            
            else: 
                #deactivate tracker due to lost
                #after this the detector has to deliver a match
                self.TrBBoxes.remove(bbox)

                
    ########
        # No foundings match...        
        Object.FramesSinceSeen += 1
        tracker.UpdateCond = False
        self.log.debug("No found_ob for Object #%d. FramesSinceSeen=%d"
            , Object.id, Object.FramesSinceSeen)

        return None




##########################        
##########################    





    def UpdateCount(self, foundings, CurrFrameNumber, binary_img, frame = None):
        """
        manages the new foundings 
        updates trackers, assignes foundings, filters not moving obects
        inits new foundings and trackers 

        Args:
            foundings ([list(contour, centroid, last_frame)]): list of found objects in frame
            CurrFrameNumber ([int]): index of current frame
            frame ([np.ndarray], optional): [description]. Defaults to None.
        Returns:
            ([list(ObjectClass)]): all so far known Objects
        """




        self.log.debug("Updating count using %d foundings...", len(foundings))

        self.CurrFrameNumber = CurrFrameNumber
        self.TrBBoxes = []            #tracker of so far known objects in current frame
        
        
        #update all tracker
        for tracker in self.Trackers:
            if tracker.UpdateCond == True:
                ret, bbox, centroid, id = tracker.Update(binary_img)
                if ret:
                    self.TrBBoxes.append((bbox, centroid, id))
                
        
        # remove const./not moving objects from foundings
        if len(self.ConstObjects) > 0:    
            for found_ob in foundings:
                contour, centroid, last_frame = found_ob
                for obj in self.ConstObjects:
                    if centroid[0]>=obj.position[0]-15 and centroid[0]<=obj.position[0]+15:
                        if centroid[1]>=obj.position[1]-15 and centroid[1]<=obj.position[1]+15:
                            foundings.remove(found_ob)
                            break
                        
            # remove also corresp. trackers
            for bbox in self.TrBBoxes:
                box, centroid, id = bbox
                for obj in self.ConstObjects:
                    if centroid[0]>=obj.position[0]-15 and centroid[0]<=obj.position[0]+15:
                        if centroid[1]>=obj.position[1]-15 and centroid[1]<=obj.position[1]+15:
                            self.TrBBoxes.remove(bbox)
                            break        


        # update all the existing Objects
        for Object in self.Objects:
            i = self.UpdateObject(Object,foundings, frame)
            if i is not None:
                del foundings[i]  # delete found_ob if assigned


        # add remaining foundings as new Objects and init trackers
        for found_ob in foundings:
            contour, centroid, last_frame = found_ob    
            new_obj = ObjectClass(self.next_vehicle_id, centroid, self.CurrFrameNumber, contour)
            self.Objects.append(new_obj)
            
            new_tracker = MultiTracker(self.next_vehicle_id, centroid, contour, binary_img)
            self.Trackers.append(new_tracker)
            
            self.next_vehicle_id += 1
            self.log.debug("Created new Object #%d from found_ob (%d, %d)."
                , new_obj.id, centroid[0], centroid[1])


        #detect and remove not moving vehicles
        for Object in self.Objects:
            positions = Object.Positions[-4:]
            if len(positions) == 4:
                x_pos = [item[0][0] for item in positions]
                y_pos = [item[0][1] for item in positions]
                if (max(x_pos)-min(x_pos))<=30 and (max(y_pos)-min(y_pos))<=30:
                    Object.Constant=True
                    for tracker in self.Trackers:
                        if Object.id == tracker.id:
                            tracker.Remove = True
                    
            if Object.Constant == True:
                #add to class storing the not movgin objects
                new_const_vehicle = Const_Object(self.next_const_vehicle_id, Object.LastPosition[0] )
                self.ConstObjects.append(new_const_vehicle)

        
        #remove const vehicles
        self.Objects[:] = [ obj for obj in self.Objects
            if obj.Constant==False]
        #remove tracker of those vehicles
        self.Trackers[:] = [ t for t in self.Trackers
            if t.remove == False]
        
        self.log.debug("Objects updated, tracking %d Objects.", len(self.Objects))

        for Object in self.Objects:
            if Object.LastVector != 0:
                Object.DriftUpdate()
                
        return  self.Objects



