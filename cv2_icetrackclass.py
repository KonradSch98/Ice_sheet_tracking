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


    def draw(self, output_image):
        car_colour = CAR_COLOURS[self.id % len(CAR_COLOURS)]
        for point in self.positions:
            cv2.circle(output_image, point, 2, car_colour, -1)
            cv2.polylines(output_image, [np.int32(self.positions)]
                , False, car_colour, 1)



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

        return distance, angle 



####################        
####################    



    @staticmethod
    def is_valid_vector(old_vector, new_vector, frames_since_seen):
        
        if type(old_vector) is tuple:
            old_distance, old_angle = old_vector
            new_distance, new_angle = new_vector
            
            if frames_since_seen == 0:
                upper_d = 1.2 * old_distance
                lower_d = .8 * old_distance
                upper_a = 1.2 * old_angle
                lower_a = 0.8 * old_angle
            if frames_since_seen <= 7 :
                #for distant vectors extent angle range by last_seen
                frames_since_seen += 1
                upper_d = 1.2**frames_since_seen * frames_since_seen*old_distance
                lower_d = 0.8**frames_since_seen * frames_since_seen*old_distance
                upper_a = 1.2**frames_since_seen * old_angle
                lower_a = 0.8**frames_since_seen * old_angle
            else: return False
                
            # if new_distance == 0 and new_angle==0:
            #     #eliminate equal pairs
            #     return False
            if new_distance <= upper_d and new_distance >= lower_d:
                if new_angle <= upper_a and new_angle >= lower_a:
                    return True
            elif new_distance<=30:
                return True
            else: return False
        
        elif new_vector[0]<= 350:
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
            
            if w_n <= w_o *1.4 and w_n >= w_o *.6:
                if h_n <= h_o *1.4 and h_n >= h_o *.6:
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
        cv2.rectangle(outputimg, (x, y), (x + w - 1, y + h - 1), (255,0,0), 2)
        cv2.circle(outputimg, centroid, 3, (0,255,0), -1)
        #ax.imshow(outputimg)
        #cv2.waitKey(0)
        
        contour, (centroid, frame_number) = match[1]
        x, y, w, h = contour

        # Mark the bounding box and the centroid on the processed frame
        # NB: Fixed the off-by one in the bottom right corner
        cv2.rectangle(outputimg, (x, y), (x + w - 1, y + h - 1), (255,0,0), 2)
        cv2.circle(outputimg, centroid, 10, (255,0,0), -1)
        
        
        
        return outputimg
 
 

 
####################        
####################    

   
    
    
    def update_vehicle(self, vehicle, matches,frame_number_act, output_image=None):
        # Find if any of the matches fits this vehicle
        for i, match in enumerate(matches):
            contour, centroid, frame_number = match

            vector = self.get_vector(vehicle.last_position[0], centroid)
            if vector != 0 and vector !=(0,0):
                if output_image is not None:
                    copy = output_image.copy()
                    frame = self.printmarker([match, (vehicle.last_contour, vehicle.last_position)], copy)
            #eliminate equal pairs      
                print("vector: " + str(self.is_valid_vector(vehicle.vector, vector, vehicle.frames_since_seen)))
                print("contour: " + str(self.is_valid_contour(vehicle.last_contour, contour)))  
                if self.is_valid_vector(vehicle.vector, vector, vehicle.frames_since_seen) and self.is_valid_contour(vehicle.last_contour, contour):
                    if vehicle.frames_since_seen > 0:
                        vehicle.vector = vehicle.vector
                    else: vehicle.vector = vector
                    
                    vehicle.add_position((centroid, frame_number_act))
                    vehicle.add_contour(contour)
                    self.log.debug("Added match (%d, %d) to vehicle #%d. vector=(%0.2f,%0.2f)"
                        , centroid[0], centroid[1], vehicle.id, vector[0], vector[1])
                    return i
            elif vector == 0 or vector ==(0,0):
                return i

        # No matches fit...        
        vehicle.frames_since_seen += 1
        self.log.debug("No match for vehicle #%d. frames_since_seen=%d"
            , vehicle.id, vehicle.frames_since_seen)

        return None



####################        
####################    




    def update_count(self, matches, frame_number_act, output_image = None):
        self.log.debug("Updating count using %d matches...", len(matches))


        # Add new vehicles based on the remaining matches
        # but only if its no constant object
        for match in matches:
            contour, centroid, frame_number = match
            if len(self.const_vehicles) > 0:
                for veh in self.const_vehicles:
                    if centroid[0]>=veh.position[0]-15 and centroid[0]<=veh.position[0]+15:
                        if centroid[1]>=veh.position[1]-15 and centroid[1]<=veh.position[1]+15:
                            matches.remove(match)
                            break

        #print_matches = matches.copy()   
                     
        # First update all the existing vehicles
        for vehicle in self.vehicles:
            i = self.update_vehicle(vehicle, matches, frame_number_act, output_image)
            if i is not None:
                del matches[i]  # delete already known figure


                
                
        for match in matches:
            contour, centroid, frame_number = match    
            new_vehicle = Vehicle(self.next_vehicle_id, centroid, frame_number_act, contour)
            self.next_vehicle_id += 1
            self.vehicles.append(new_vehicle)
            self.log.debug("Created new vehicle #%d from match (%d, %d)."
                , new_vehicle.id, centroid[0], centroid[1])



        #remove not moving vehicles
        for vehicle in self.vehicles:

            positions = vehicle.positions[-3:]
            x_pos = [item[0][0] for item in positions]
            y_pos = [item[0][1] for item in positions]
            if len(positions) == 3:
                if (max(x_pos)-min(x_pos))<=15 and (max(y_pos)-min(y_pos))<=15:
                    vehicle.Constant=True
                    

            if vehicle.Constant == True:
                new_const_vehicle = Const_Vehicle(self.next_const_vehicle_id, vehicle.last_position[0] )
                self.const_vehicles.append(new_const_vehicle)



        # # Optionally draw the vehicles on an image
        # if output_image is not None:
        #     for vehicle in self.vehicles:
        #         vehicle.draw(output_image)

            cv2.putText(output_image, ("%02d" % self.vehicle_count), (142, 10)
                , cv2.FONT_HERSHEY_PLAIN, 0.7, (127, 255, 255), 1)




        # Remove vehicles that have not been seen long enough
        removed = [ v.id for v in self.vehicles
            if v.frames_since_seen >= self.max_unseen_frames ]
        self.vehicles[:] = [ v for v in self.vehicles
            if v.Constant==False]
        for id in removed:
            self.log.debug("Removed vehicle #%d.", id)

        self.log.debug("Count updated, tracking %d vehicles.", len(self.vehicles))




        # for vehicle in self.vehicles: 

        pos = [item[1] for item in matches]
        if vehicle.last_position[0] not in pos:
            matches.append((vehicle.last_contour,vehicle.last_position[0], vehicle.last_position[1]))
        
        return matches, self.vehicles



