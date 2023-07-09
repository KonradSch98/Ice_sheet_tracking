import logging
import math

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TKAgg')






CAR_COLOURS = [ (0,0,255), (0,106,255), (0,216,255), (0,255,182), (0,255,76)
    , (144,255,0), (255,255,0), (255,148,0), (255,0,178), (220,0,255) ]



#################################################################################




class MultiTracker(object):
    
    def __init__(self, id, position, frame_number, contour, frame) -> None:
        self.id = id
        
        self.tracker = cv2.legacy.TrackerCSRT_create()
        self.remove = False
        ret = self.tracker.init(frame, contour)
        if not ret:
            print('cannot init tracker of id %i' % self.id)
            return None
        self.position = [0,0]
        self.position[1] = position
        self.contour = [0,0]        
        self.contour[1] = contour
        self.UpdateCond = True
        
    def update(self, frame):
        ret, bbox = self.tracker.update(frame)

        bbox = [int(i) for i in bbox]
        (x, y, w, h) = bbox
        x1 = int(w / 2)
        y1 = int(h / 2)
        cx = x + x1
        cy = y + y1
        centroid =  (int(cx), int(cy))
        self.position[0] = self.position[1]
        self.position[1] = centroid
        self.contour[0] = self.contour[1]
        self.contour[1] = bbox
        
        return ret, bbox, centroid, self.id



class Vehicle(object):
    def __init__(self, id, position, frame_number, contour):
        self.id = id
        self.positions = [(position, frame_number)]
        #self.positions = property()
        self.frames_since_seen = 0
        self.counted = False
        self.Constant = False
        self.contours = [contour]
        self.vector = 0
        self.av_vector = np.zeros(2,list)

        
    @property
    def last_position(self):
        return self.positions[-1]

    @property
    def last_contour(self):
        return self.contours[-1]

    
    def add_position(self, new_position):
        self.positions.append(new_position)
        self.frames_since_seen = 0
        
    def add_contour(self, new_contour):
        self.contours.append(new_contour)

    def drift_update(self):
        drift_n = self.av_vector[0]
        index_n = self.av_vector[1]
        
        vector_n1 = np.array(self.vector)
        offset = ( vector_n1 -  drift_n) / (index_n + 1)
        #update drift
        self.av_vector[0] += offset
        self.av_vector[1] += 1
        



#################################################################################



class Const_Vehicle(object):
    def __init__(self, id, position):
        self.id = id
        self.position = position




#################################################################################




class VehicleCounter(object):
    def __init__(self, shape, divider):
        self.log = logging.getLogger("vehicle_counter")

        self.height, self.width = shape
        self.divider = divider

        self.vehicles = []
        self.trackers = []
        self.const_vehicles = []
        self.next_vehicle_id = 0
        self.next_const_vehicle_id = 0
        self.vehicle_count = 0
        self.max_unseen_frames = 7


    @staticmethod
    def get_vector(a, b):
        """Calculate vector (distance, angle in degrees) from point a to point b.

        Angle ranges from -180 to 180 degrees.
        Vector with angle 0 points straight down on the image.
        Values increase in clockwise direction.
        """
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])

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



    @staticmethod
    def is_valid_vector(vehicle, new_vector, contour, tracker=None):
        
        old_vector = vehicle.vector
        frames_since_seen = vehicle.frames_since_seen
        
        
        if type(old_vector) is tuple:
            old_distance, old_angle = old_vector
            new_distance, new_angle = new_vector
            
            if new_angle < 0 and old_angle < 0:
                new_angle = -1*new_angle
                old_angle = -1*old_angle
            
            if frames_since_seen == 0:
                upper_d = (old_distance + abs(contour[2]-vehicle.contours[-1][2])/3 + abs(vehicle.contours[-1][2]-vehicle.contours[-2][2])/3 ) *1.3
                lower_d = (old_distance - abs(contour[2]-vehicle.contours[-1][2])/3 - abs(vehicle.contours[-1][2]-vehicle.contours[-2][2])/3 ) *.7                
                # upper_d = 1.4 * old_distance
                # lower_d = .6 * old_distance
                upper_a = 1.2 * old_angle
                lower_a = 0.8 * old_angle
            elif frames_since_seen <= 2 :
                #for distant vectors extent angle range by last_seen
                frames_since_seen += 1
                upper_d = 1.4**frames_since_seen * frames_since_seen*old_distance
                lower_d = 0.6**frames_since_seen * frames_since_seen*old_distance
                upper_a = 1.2**frames_since_seen * old_angle
                lower_a = 0.8**frames_since_seen * old_angle
            else: return False
            
            if tracker == 'tracker':
                if new_distance <= upper_d and new_distance >= lower_d:
                    return True
            else:     
                if new_distance <= upper_d and new_distance >= lower_d:
                    if new_angle <= upper_a and new_angle >= lower_a:
                        return True
            if new_distance<=30:
                return True
            else: return False
        
        elif new_vector[0]<= 300:
            #initiate by closeness
            return True
        
        else: return False
            # if 
            # threshold_distance = max(10.0, -0.008 * angle**2 + 0.4 * angle + 30.0)
            # return (distance <= threshold_distance)




####################        
####################    




    @staticmethod
    def is_valid_contour(old_contour, new_contour):
    
        if type(old_contour) is tuple:
            x_o, y_o, w_o, h_o = old_contour
            x_n, y_n, w_n, h_n = new_contour
            
            if w_n <= w_o *1.5 and w_n >= w_o *.5:
                if h_n <= h_o *1.5 and h_n >= h_o *.5:
                    return True
                else: return False
            else: return False
            
        # else:
        #     if 
        #     threshold_distance = max(10.0, -0.008 * angle**2 + 0.4 * angle + 30.0)
        #     return (distance <= threshold_distance)
    
 
 
####################        
####################    

   
    
    def printmarker(self,match, outputimg):
        
        #for (i, match) in enumerate(matches):
        contour, centroid, frame_number = match[0]
        x, y, w, h = contour

        # Mark the bounding box and the centroid on the processed frame
        # NB: Fixed the off-by one in the bottom right corner
        cv2.rectangle(outputimg, (x-6, y-6), (x + w - 6, y + h - -6), (0,255,0), 2)
        cv2.circle(outputimg, centroid, 3, (0,255,0), -1)
        cv2.putText(outputimg, 'bbox', fontFace = cv2.FONT_HERSHEY_SIMPLEX, color = (0,0,0), org = (centroid[0]+20, centroid[1]+20), fontScale = 3, thickness = 2)
        #ax.imshow(outputimg)
        #cv2.waitKey(0)
        
        contour, centroid, frame_number = match[1]
        x, y, w, h = contour

        # Mark the bounding box and the centroid on the processed frame
        # NB: Fixed the off-by one in the bottom right corner
        cv2.rectangle(outputimg, (x-6, y-6), (x + w - 6, y + h - -6), (255,0,0), 2)
        cv2.circle(outputimg, centroid, 3, (255,0,0), -1)
        cv2.putText(outputimg, 'match', fontFace = cv2.FONT_HERSHEY_SIMPLEX, color = (0,0,0), org = (centroid[0]+20, centroid[1]+20), fontScale = 3, thickness = 2)
        #ax.imshow(outputimg)
        #cv2.waitKey(0)
        
     
        
        contour, (centroid, frame_number) = match[2]
        x, y, w, h = contour

        # Mark the bounding box and the centroid on the processed frame
        # NB: Fixed the off-by one in the bottom right corner
        cv2.rectangle(outputimg, (x, y), (x + w - 1, y + h - 1), (255,0,0), 2)
        cv2.circle(outputimg, centroid, 10, (255,0,0), -1)
        cv2.putText(outputimg, 'l_veh', fontFace = cv2.FONT_HERSHEY_SIMPLEX, color = (0,0,0), org = (centroid[0]+20, centroid[1]+20), fontScale = 3, thickness = 2)
        
        contour, centroid = match[3]
        x, y, w, h = contour

        # Mark the bounding box and the centroid on the processed frame
        # NB: Fixed the off-by one in the bottom right corner
        cv2.rectangle(outputimg, (x, y), (x + w - 1, y + h - 1), (0,255,0), 2)
        cv2.circle(outputimg, centroid, 10, (0,255,0), -1)
        cv2.putText(outputimg, 'l_track', fontFace = cv2.FONT_HERSHEY_SIMPLEX, color = (0,255,0), org = (centroid[0]+20, centroid[1]+20), fontScale = 3, thickness = 2)
        
        
        
        return outputimg
 
 

 
####################        
####################    

   
    
    
    def update_vehicle(self, vehicle, matches, output_image=None):
        # Find if any of the matches fits this vehicle

        # if len(vehicle.positions) == 1:
        #     id = vehicle.id
        #     #get correpsonding tracker bbox
        #     for bbox in self.bboxes:
        #         if bbox[2] == id:
        #             break
            
        #     #look if match equals tracker bbox
        #     for i, match in enumerate(matches):
        #         contour, centroid, frame_number = match
        #         vector = self.get_vector(bbox[1], centroid)
        #         #if not add bbox
        #         #if yes add match remove match
        #         if vector[0] <= 60:
        #             contour, centroid, frame_number = match
        #             vector = self.get_vector(vehicle.last_position[0], centroid)
        #             vehicle.vector = vector   
        #             #update bboxes
        #             for bbox in self.bboxes:
        #                 if bbox[2] == id:
        #                     break
                            
        #             for tracker in self.trackers:
        #                 if tracker.id == id:
        #                     break
                        
        #             #reset tracker
        #             tracker.remove = True
        #             reset_tracker = MultiTracker(id, centroid, self.frame_number_act, contour, output_image)
        #             self.trackers.append(reset_tracker)
                    
        #             copy = output_image.copy()
        #             frame = self.printmarker([bbox, match, (vehicle.last_contour, vehicle.last_position), (tracker.contour[0], tracker.position[0])], copy)
        #             self.bboxes.remove(bbox)

        #             #update vehicle
        #             vehicle.add_position((centroid, self.frame_number_act))
        #             vehicle.add_contour(contour)
        #             return i

                
        #     centroid = bbox[1]
        #     contour = tuple(bbox[0])
        #     vector = self.get_vector(vehicle.last_position[0], centroid)
        #     if vehicle.frames_since_seen > 0:
        #         vehicle.vector = vehicle.vector
        #     else: vehicle.vector = vector
            
        #     for tracker in self.trackers:
        #         if tracker.id == id:
        #             break
        #     copy = output_image.copy()
        #     frame = self.printmarker([bbox, bbox, (vehicle.last_contour, vehicle.last_position), (tracker.contour[0], tracker.position[0])], copy)
            
        #     #update vehicle
        #     vehicle.add_position((centroid, self.frame_number_act))
        #     vehicle.add_contour(contour)  
        #     self.bboxes.remove(bbox)
        #     return None


        
        # else:
        bbox = None
        tracker = None
        id = vehicle.id
        
        #find corresponding bboxes / tracker
        for bbox in self.bboxes:
            if bbox[2] == id:
                break
        for tracker in self.trackers:
            if tracker.id == id:
                break
                        

        for i, match in enumerate(matches):
            contour, centroid, frame_number = match
            
        
            vector = self.get_vector(vehicle.last_position[0], centroid)
            #eliminate equal pairs
            if vector != 0 and vector !=(0,0):

                    
                val_vec = self.is_valid_vector(vehicle, vector, contour)
                val_con = self.is_valid_contour(vehicle.last_contour, contour)

                if  val_vec and val_con:
                

                    if bbox is not None:
                        copy = output_image.copy()
                        frame = self.printmarker([bbox, match, (vehicle.last_contour, vehicle.last_position), (tracker.contour[0], tracker.position[0])], copy)

                    
                    
                    if vehicle.frames_since_seen > 0:
                        vehicle.vector = (vector[0]/(vehicle.frames_since_seen+1),vector[1])
                    else: vehicle.vector = vector
                    
                        
                    #update vehicle
                    vehicle.add_position((centroid, self.frame_number_act))
                    vehicle.add_contour(contour)
                    
                    
                    #########################
                    #reset tracker if lost
                    if bbox is None:
                        tracker.remove = True
                        reset_tracker = MultiTracker(id, centroid, self.frame_number_act, contour, output_image)
                        self.trackers.append(reset_tracker)
                    else:
                        vector = self.get_vector(bbox[1], centroid)
                        #reset tracker if bad
                        if vector[0] > 50:
                            tracker.remove = True
                            reset_tracker = MultiTracker(id, centroid, self.frame_number_act, contour, output_image)
                            self.trackers.append(reset_tracker) 
                        else: tracker.UpdateCond = True 
                    
                    if bbox is not None:
                        self.bboxes.remove(bbox)  
                                          
                    self.log.debug("Added match (%d, %d) to vehicle #%d. vector=(%0.2f,%0.2f)"
                        , centroid[0], centroid[1], vehicle.id, vector[0], vector[1])
                    
   
                    return i
                
                
                
            elif vector == 0 or vector ==(0,0):
                vehicle.frames_since_seen += 1
                
                #reset tracker for constant object ???
                tracker.remove = True
                reset_tracker = MultiTracker(id, centroid, self.frame_number_act, contour, output_image)
                self.trackers.append(reset_tracker)
                if bbox is not None:
                    self.bboxes.remove(bbox)  
                
                return i



        #if no detector found, add tracker
        if bbox is not None:

            centroid = bbox[1]
            contour = tuple(bbox[0])

            vector = self.get_vector(vehicle.last_position[0], centroid)
            
            #check if tracker is doing good
            val_vec = self.is_valid_vector(vehicle, vector, contour, 'tracker')
            val_con = self.is_valid_contour(vehicle.last_contour, contour)
            
            if  val_vec and val_con:
                
                copy = output_image.copy()
                frame = self.printmarker([bbox, bbox, (vehicle.last_contour, vehicle.last_position), (tracker.contour[0], tracker.position[0])], copy)
                for i, match in enumerate(matches):
                    contour, centroid, frame_number = match
                    x, y, w, h = contour
                    cv2.rectangle(frame, (x, y), (x + w - 3, y + h - 3), (0,2,255), 2)
                
                
                centroid = bbox[1]
                contour = tuple(bbox[0])
                
                if vehicle.frames_since_seen > 0:
                    vehicle.vector = (vector[0]/(vehicle.frames_since_seen+1),vector[1])
                else: vehicle.vector = vector 
                
                #update vehicle
                vehicle.add_position((centroid, self.frame_number_act))
                vehicle.add_contour(contour) 
                
                
                
                self.bboxes.remove(bbox)
                tracker.UpdateCond = True
                return None
            
            else: 
                tracker.UpdateCond = False
                self.bboxes.remove(bbox)

                

        # No matches fit...        
        vehicle.frames_since_seen += 1
        tracker.UpdateCond = False
        self.log.debug("No match for vehicle #%d. frames_since_seen=%d"
            , vehicle.id, vehicle.frames_since_seen)

        return None



####################        
####################    




    def update_count(self, matches, frame_number_act, frame, output_image = None):
        self.log.debug("Updating count using %d matches...", len(matches))

        self.frame_number_act = frame_number_act
        self.bboxes = []
        
        
        #first update all tracker
        for tracker in self.trackers:
            if tracker.UpdateCond == True:
                ret, bbox, centroid, id = tracker.update(frame)
                if ret:
                    self.bboxes.append((bbox, centroid, id))
                
            
        if len(self.const_vehicles) > 0:    
            # remove const. vehicles from matches
            for match in matches:
                contour, centroid, frame_number = match
                for veh in self.const_vehicles:
                    if centroid[0]>=veh.position[0]-15 and centroid[0]<=veh.position[0]+15:
                        if centroid[1]>=veh.position[1]-15 and centroid[1]<=veh.position[1]+15:
                            matches.remove(match)
                            break
                        
            # remove const. vehicles from bboxes
            for bbox in self.bboxes:
                box, centroid, id = bbox
                for veh in self.const_vehicles:
                    if centroid[0]>=veh.position[0]-15 and centroid[0]<=veh.position[0]+15:
                        if centroid[1]>=veh.position[1]-15 and centroid[1]<=veh.position[1]+15:
                            self.bboxes.remove(bbox)
                            break

        #print_matches = matches.copy() 
        
          
        
        
        
         
        # First update all the existing vehicles
        for vehicle in self.vehicles:
            i = self.update_vehicle(vehicle, matches, output_image)
            if i is not None:
                del matches[i]  # delete already known figure


                            
                
                
        # add remaining matches as new vehicles and trackers
        for match in matches:
            contour, centroid, frame_number = match    
            new_vehicle = Vehicle(self.next_vehicle_id, centroid, self.frame_number_act, contour)
            self.vehicles.append(new_vehicle)
            
            new_tracker = MultiTracker(self.next_vehicle_id, centroid, self.frame_number_act, contour, frame)
            self.trackers.append(new_tracker)
            
            self.next_vehicle_id += 1
            self.log.debug("Created new vehicle #%d from match (%d, %d)."
                , new_vehicle.id, centroid[0], centroid[1])


        
        
        

        #remove not moving vehicles
        for vehicle in self.vehicles:

            positions = vehicle.positions[-4:]
            x_pos = [item[0][0] for item in positions]
            y_pos = [item[0][1] for item in positions]
            if len(positions) == 4:
                if (max(x_pos)-min(x_pos))<=30 and (max(y_pos)-min(y_pos))<=30:
                    vehicle.Constant=True
                    for tracker in self.trackers:
                        if vehicle.id == tracker.id:
                            tracker.Remove = True
                            break
                    

            if vehicle.Constant == True:
                new_const_vehicle = Const_Vehicle(self.next_const_vehicle_id, vehicle.last_position[0] )
                self.const_vehicles.append(new_const_vehicle)



 


        # # Remove vehicles that have not been seen long enough
        # removed = [ v.id for v in self.vehicles
        #     if v.frames_since_seen >= self.max_unseen_frames ]
        
        #remove const vehicles
        self.vehicles[:] = [ v for v in self.vehicles
            if v.Constant==False]
        #remove tracker of those vehicles
        self.trackers[:] = [ t for t in self.trackers
            if t.remove == False]
        
        # for id in removed:
        #     self.log.debug("Removed vehicle #%d.", id)

        self.log.debug("Count updated, tracking %d vehicles.", len(self.vehicles))


        for vehicle in self.vehicles:
            if vehicle.vector != 0:
                vehicle.drift_update()
            
        

        #make list for printing on frame
        matches = []
        for vehicle in self.vehicles:
            if vehicle.last_position[1] == self.frame_number_act:
                matches.append((vehicle.last_contour, vehicle.last_position[0], vehicle.last_position[1]))
            
        return  self.vehicles



