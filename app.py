
import cv2
import mediapipe as mp
import numpy as np


# Initialize Mediapipe solutions
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils



# Here, right after your imports
def adjust_landmarks_for_rotation(landmarks, frame_width, frame_height):
    adjusted_landmarks = []
    for landmark in landmarks.landmark:
        adjusted_x = landmark.y
        adjusted_y = 1 - landmark.x  # Subtract from 1 because the origin is at the top left corner
        adjusted_landmarks.append(landmark)  
    return adjusted_landmarks

#rotate video for vertical 16k display
def rotate_frame(frame, rotation=cv2.ROTATE_90_CLOCKWISE):
    return cv2.rotate(frame, rotation)


#appling a moving average filter to the overlay's position
class SmoothOverlay:
    def __init__(self, buffer_size=3):
        self.buffer_size = buffer_size
        self.positions_x = []
        self.positions_y = []
        self.sizes_w = []
        self.sizes_h = []

    def update(self, x, y, w, h):
        if len(self.positions_x) >= self.buffer_size:
            self.positions_x.pop(0)
            self.positions_y.pop(0)
            self.sizes_w.pop(0)
            self.sizes_h.pop(0)

        self.positions_x.append(x)
        self.positions_y.append(y)
        self.sizes_w.append(w)
        self.sizes_h.append(h)

        smooth_x = sum(self.positions_x) / len(self.positions_x)
        smooth_y = sum(self.positions_y) / len(self.positions_y)
        smooth_w = sum(self.sizes_w) / len(self.sizes_w)
        smooth_h = sum(self.sizes_h) / len(self.sizes_h)

        return int(smooth_x), int(smooth_y), int(smooth_w), int(smooth_h)
    



# Function to convert white pixels in the video frame to transparent
def white_to_transparent(frame_video):
    if frame_video.shape[2] == 3:
        frame_video = cv2.cvtColor(frame_video, cv2.COLOR_BGR2BGRA)
    
    white_threshold = 220  # Define a threshold to consider a pixel "white"
    white_mask = np.all(frame_video[:, :, :3] > white_threshold, axis=2)  # Mask for white pixels
    frame_video[white_mask, 3] = 0  # Set alpha to 0 for white pixels, making them transparent
    
    return frame_video


# Helper function to resize the overlay image with aspect ratio preservation
def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        ratio = height / float(h)
        dimension = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dimension = (width, int(h * ratio))

    return cv2.resize(image, dimension, interpolation=inter)

# Function to overlay lungs on the person
def overlay_lungs(frame_webcam, landmarks, frame_video):
    frame_height, frame_width, _ = frame_webcam.shape

    crop_height = 10  # Number of pixels to crop from the bottom
    if frame_video.shape[0] > crop_height:
       frame_video = frame_video[:-crop_height, :, :]

    # Predefine overlay_x and overlay_y to ensure they have values
    overlay_x, overlay_y = 0,50  # Default values, adjust as necessary
    

    min_overlay_y_distance = 100  # Minimaler Abstand des Overlays von der obersten Gesichtslandmarke
       
        # Definieren Sie einen festen Abstand zwischen dem unteren Punkt der Schultern und dem oberen Rand des Overlays
    overlay_distance_from_shoulder = 30  # Diesen Wert können Sie anpassen


    if landmarks:
        # Extract  relevant landmarks
        shoulder_left = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        shoulder_right = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        hip_left = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
        hip_right = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
        face_top = landmarks[mp.solutions.pose.PoseLandmark.NOSE.value]
 
        face_top_y = int(face_top.y * frame_webcam.shape[0]) 
        #bottom shoulder point
        shoulder_bottom_y = max(shoulder_left.y, shoulder_right.y) * frame_webcam.shape[0]

        top_center_x = int(frame_webcam.shape[1] * (shoulder_left.x + shoulder_right.x) / 2)
        top_center_y = int(frame_webcam.shape[0] * (shoulder_left.y + shoulder_right.y) / 2)

        hip_center_x = int(frame_webcam.shape[1] * (hip_left.x + hip_right.x) / 2)
        hip_center_y = int(frame_webcam.shape[0] * (hip_left.y + hip_right.y) / 2)
        # use shoulder distance to calculate overlay
        overlay_height = abs(shoulder_left.y - hip_left.y) * frame_height
        overlay_width = abs(shoulder_left.x - shoulder_right.x) * frame_width
        # calculate scaling factor based on body pose 
       
        
        body_tilt_factor = (hip_center_y - shoulder_bottom_y) / frame_webcam.shape[0]

        #  Overlay-Position, shoulder distance        
        dx = shoulder_right.x - shoulder_left.x
        dy = shoulder_right.y - shoulder_left.y
        angle = np.arctan2(dy, dx)
        overlay_x_offset =  angle * -5  # Experimentieren Sie mit dem Faktor

      

       
       # Berechnen Sie die neue Y-Position des Overlays
        overlay_y = int(shoulder_bottom_y + overlay_distance_from_shoulder)
      
        overlay_y = max(face_top_y + min_overlay_y_distance, overlay_y)
       
       
       
        shoulder_distance = np.linalg.norm(np.array([shoulder_left.x, shoulder_left.y]) - np.array([shoulder_right.x, shoulder_right.y]))

        distance_vertical = np.linalg.norm(np.array([top_center_x, top_center_y]) - np.array([hip_center_x, hip_center_y]))
        scale_factor = max(1.2 , 1 - (distance_vertical * 0.01))   # Justieren Sie den Faktor
       
        p_width_multiplier = scale_factor
        p_height_multiplier = 0.7 * scale_factor
        # Anwendung der Skalierung und Offset
        # Basierende Berechnung von width und height
        p_width = int(shoulder_distance * frame_webcam.shape[1] * p_width_multiplier)
        p_height = int(p_height_multiplier * (hip_center_y - top_center_y))

        # Feinjustierung basierend auf body_tilt_factor
        height_scaling_factor = max(0.9, 1 - body_tilt_factor)
        width_scaling_factor = height_scaling_factor

        width = int(p_width * width_scaling_factor)
        height = int(p_height * height_scaling_factor)


        smooth_overlay = SmoothOverlay(buffer_size=5)  # Adjust buffer_size as needed

       # Inside your overlay_lungs function or where you set the overlay position:
        overlay_x, overlay_y, width, height = smooth_overlay.update(overlay_x, overlay_y, width, height)
        
     
        # Create Alpha_channel
        if frame_video.shape[2] == 3:
            alpha_channel = np.ones((frame_video.shape[0], frame_video.shape[1], 1), dtype=frame_video.dtype) * 220
            frame_video_with_alpha = np.concatenate((frame_video, alpha_channel), axis=-1)
            frame_video_with_alpha = white_to_transparent(frame_video_with_alpha)
            
        else:
           frame_video_with_alpha = white_to_transparent(frame_video_with_alpha)

           
        # Calculate overlay position with a downward offset, ensuring it's within frame bounds
        overlay_x = max(top_center_x - width // 2 + int(overlay_x_offset), 0)
        
       
        end_x = min(overlay_x + width, frame_webcam.shape[1])
        end_y = min(overlay_y + height, frame_webcam.shape[0])


       
    
        # Adjust the overlay size (if necessary)
        overlay_width = end_x - overlay_x
        overlay_height = end_y - overlay_y

    if overlay_width > 0 and overlay_height > 0:
        # Resize the video frame with alpha to precisely match the overlay dimensions
        lungs_resized_adjusted = cv2.resize(frame_video_with_alpha, (overlay_width, overlay_height))

        # Perform alpha blending
        alpha = lungs_resized_adjusted[:, :, 3] / 220.0
        for c in range(3):  # Iterate over RGB channels
            target_section = frame_webcam[overlay_y:end_y, overlay_x:end_x, c]
            frame_webcam[overlay_y:end_y, overlay_x:end_x, c] = (
                alpha * lungs_resized_adjusted[:, :, c] + (1 - alpha) * target_section
            )
        else:
          print("Invalid overlay dimensions:", overlay_width, "x", overlay_height)

    return frame_webcam

# Function for Process_video
def process_video(webcam_index, video_path):
    # Initialize video capture for the webcam and the overlay video
    cap_webcam = cv2.VideoCapture(webcam_index)
    cap_video = cv2.VideoCapture(video_path)

    # Initialize Mediapipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence= 0.5, min_tracking_confidence= 0.7)

    while cap_webcam.isOpened():
        success_webcam, frame_webcam = cap_webcam.read()
        if success_webcam:
        # Rotate the frame for 9:16 orientation
           frame_webcam = rotate_frame(frame_webcam)

           if not success_webcam:
              break


        success_video, frame_video = cap_video.read()
       
        if not success_webcam:
            print("Ignoring empty camera frame.")
            continue

        
        if not success_video:
            cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video
            success_video, frame_video = cap_video.read()
        
        if not  success_video:
            print("Failed to loop video.")
            continue
          # Extrahieren der Frame-Dimensionen
 
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

         # Process the frame with Mediapipe Pose
        results = pose.process(frame_rgb)


   
       
        frame_rgb.flags.writeable = True
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Check if pose landmarks were detected and overlay lungs
        if results.pose_landmarks:
          

            frame_webcam = overlay_lungs(frame_webcam, results.pose_landmarks.landmark, frame_video)
       
            
       

        # Display  frame
        cv2.imshow('Lungs Overlay', frame_webcam)

                
        # Break  loop if 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
           break
  

    # Release resources
    cap_webcam.release()
    cap_video.release()
    cv2.destroyAllWindows()


def overlay_image(background, overlay, x, y):

    return background
process_video(0, "beta.mov")

