"""
A game that uses hand tracking to 
hit and destroy green circle enemies.

@author: Nandhini Namasivayam
@version: March 2024

edited from: https://i-know-python.com/computer-vision-game-using-mediapipe-and-python/
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import random
import time
import numpy as np

# STEP 2: Create an PoseLandmarker object.
# base_options = python.BaseOptions(model_asset_path='data/pose_landmarker.task')
# options = vision.PoseLandmarkerOptions(
#     base_options=base_options,
#     output_segmentation_masks=True)
# detector = vision.PoseLandmarker.create_from_options(options)

# # STEP 3: Load the input image.
# image = mp.Image.create_from_file("image.jpg")

# # STEP 4: Detect pose landmarks from the input image.
# detection_result = detector.detect(image)

# STEP 5: Process the detection result. In this case, visualize it.
# annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
# cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Library Constants

VisionRunningMode = mp.tasks.vision.RunningMode
DrawingUtil = mp.solutions.drawing_utils

class Lines:
    """
    A class to represent a random circle
    enemy. It spawns randomly within 
    the given bounds.
    """
    def __init__(self, color, screen_width=600, screen_height=400):
        self.color = color
        self.screen_width = screen_width
        self.screen_height = screen_height
        
    
      
class Game:
    def __init__(self):
        # Create the hand detector
        base_options = python.BaseOptions(model_asset_path='data/pose_landmarker_full.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True)
        self.detector = vision.PoseLandmarker.create_from_options(options)

        # Load video
        self.video = cv2.VideoCapture(0)
        
        #Conditions of pull-up
        self.pull_up_count = 0
        self.pull_up_valid = True
        self.pull_up_started = False
        self.feet_kicking = False
        self.swaying = False
    #Add to the pullup count variable if it is valid
    def add_to_pull_up_count(self, pose_landmarks):
        if((pose_landmarks[15].y > pose_landmarks[7].y) and (pose_landmarks[16].y > pose_landmarks[8].y) and self.pull_up_valid):
            self.pull_up_count += 1
            self.pull_up_started = False
            print(self.pull_up_count)
    #Boolean when the pull up starts
    def when_pull_up_starts(self, pose_landmarks):
        if((pose_landmarks[16].y < pose_landmarks[6].y) and (pose_landmarks[15].y< pose_landmarks[3].y)):
            self.pull_up_started = True
    #Boolean if kicking feet
    def decide_kicking_feet(self, pose_landmarks):
        if(self.pull_up_started and ((abs(pose_landmarks[28].z - pose_landmarks[24].z) > 0.1) or (abs(pose_landmarks[27].z - pose_landmarks[23].z) > 0.1))):
            self.feet_kicking = True
        else:
            self.feet_kicking = False
    #Boolean if swaying
    def decide_swaying(self, pose_landmarks):
        if((abs(pose_landmarks[12].y - pose_landmarks[11].y) > 0.05)):
            self.swaying = True
        else:
            self.swaying = False
    #Boolean is pull up valid
    def decide_pull_up_valid(self, pose_landmarks):
        if(self.swaying or self.feet_kicking):
            self.pull_up_valid = False
            return False
        else:
            self.pull_up_valid = True
            return True
    
    def get_pose_landmarks(self, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        # annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]
            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])

            return pose_landmarks
        return None

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]
            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
            
            

            #Check and add
            self.when_pull_up_starts(pose_landmarks)
            if(self.pull_up_started and self.decide_pull_up_valid(pose_landmarks)):
                self.add_to_pull_up_count(pose_landmarks)
        return annotated_image

    def run(self):
        """
        Main game loop. Runs until the 
        user presses "q".
        """    
        # TODO: Modify loop condition  
        while True:
            # Get the current frame
            frame = self.video.read()[1]

            # Convert it to an RGB image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #The image comes mirrored - flip
            image = cv2.flip(image, 1)

            # Convert the image to a readable format and find the hands
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)
            image = self.draw_landmarks_on_image(image, results)

            cv2.putText(image,
                        str(self.pull_up_count),
                        (50, 80),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=2,
                        color=GREEN,
                        thickness=3)
            
            #Making green or red circle on screen
            
            pose_landmarks = self.get_pose_landmarks(results)
            if pose_landmarks:
                self.decide_kicking_feet(pose_landmarks)
                self.decide_swaying(pose_landmarks)

                if(self.decide_pull_up_valid(pose_landmarks)):
                    cv2.circle(image,
                            (1100, 360),
                            20,
                            GREEN,
                            35)
                if(self.decide_pull_up_valid(pose_landmarks) == False):
                    cv2.circle(image,
                            (1100, 360),
                            20,
                            RED,
                            35)
                
            # Change the color of the frame back
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow('Pose Tracking', image)
            # Break the loop if the user presses 'q'
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
        self.video.release()
        cv2.destroyAllWindows()
        


if __name__ == "__main__":        
    g = Game()
    g.run()