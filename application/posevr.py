# external modules
from cv2 import cvtColor, COLOR_RGB2BGR
import time
import mediapipe as mp
from threading import Thread
from PIL import Image

from camera import get_camera_image, close_camera, get_camera
from ui import init_tkinter_app, update_video
from pipe import close_pipe, create_pipe, send_data_to_pipe, start_pipe

# setup mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.8,min_tracking_confidence=0.8,smooth_landmarks=True) 

# recorded positions
POSITIONS = ['LEFT_HIP','RIGHT_HIP','LEFT_ANKLE','RIGHT_ANKLE','LEFT_WRIST','RIGHT_WRIST', 'NOSE', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
HEEL_HEIGHT = .8
HAND_HEIGHT = .3
MIN_VISIBILITY = 0.7

prev_time = 0
prev_landmarks = {}


def video_stream_loop():
    while True:
        process_camera_image()
        time.sleep(0.01)

def process_camera_image() -> Image:
    cap = get_camera()
    image = get_camera_image(cap)
    results = pose.process(image)        
    data = convert_to_pipe_data(results)
    send_data_to_pipe(data)
    image.flags.writeable = True
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    image = cvtColor(image, COLOR_RGB2BGR)
    update_video(image)

def convert_to_pipe_data(results):
    global prev_landmarks
    global prev_time

    send = 'n'

    if results.pose_landmarks:
        landmarks = {}
        for position in POSITIONS:
            r = results.pose_landmarks.landmark[getattr(mp_pose.PoseLandmark, position)]
            if not r or (r.visibility) < MIN_VISIBILITY:
                if position in prev_landmarks:
                    landmarks[position] = prev_landmarks[position]
                else:
                    landmarks[position] = (0, 0, 0) # only happens at very beginning if landmarks occluded
            else:
                landmarks[position] = (r.x, r.y, r.z)
        
        cur_time = time.time() - prev_time
        prev_time = time.time()

        send = f'{round(cur_time, 5)};'
        for landmark, coord in landmarks.items():
            send += f'{landmark};{round(coord[0], 5)};{round(coord[1], 5)};{round(coord[2], 5)};'

        prev_landmarks = landmarks
    return send

if __name__ == '__main__':
    pipe = create_pipe()
    
    pipe_thread = Thread(target=start_pipe, args=(pipe,))
    pipe_thread.start()

    camera_thread = Thread(target=video_stream_loop)
    camera_thread.start()
    
    prev_time = time.time()

    root = init_tkinter_app()
    root.mainloop()

    quit = True

    close_pipe()
    close_camera()
