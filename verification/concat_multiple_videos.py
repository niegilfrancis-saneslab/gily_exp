import glob
import os
import tqdm
from collections import defaultdict
import re
import numpy as np
import cv2

# Functions to sort file names correctly
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

experiment_no = 492
index = 20

search_str = "%03d"%(index)

# searching for video files
video_files = glob.glob(f"D:/big_setup/experiment_{experiment_no}/concatenated_data_cam_mic_sync/video_*_{search_str}.mp4")
video_files.sort(key=natural_keys)


# resizing your video frame
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

# getting all the videos in opencv
vid_streams = []
for i in video_files:
    vid_streams.append(cv2.VideoCapture(i))

# writing into a video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f"D:/sync_test_videos/exp_{experiment_no}_{search_str}.mp4", fourcc, 30.0, (1200,600), isColor=False) # the damn shape is the opposite of the array shape

# flag to indicate if one of the camera streams os done
eof_flag = 0

while True: 
    frames = []
    for i in vid_streams:
        success, frame = i.read()
        if not success:
            eof_flag = 1
            break
        else:
            frames.append(rescale_frame(frame,25))
        
    if eof_flag: 
        print("End of video reached")
        break
    else:
        # convert to grayscale 
        gr_frames = []
        for i in frames:
            gr_frames.append(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY))
        # concatenating the frames
        frame_1 = cv2.hconcat([gr_frames[3],gr_frames[2],gr_frames[0]])
        frame_2 = cv2.hconcat([gr_frames[4],gr_frames[5],gr_frames[1]])

        final_frame = cv2.vconcat([frame_1, frame_2])
        # print(final_frame.shape)

        out.write(final_frame)
        cv2.imshow("Output", final_frame)
        if cv2.waitKey(1) & 0XFF == ord('q'): # checking if q is pressed
            print("Quitting")
            break
        
for i in vid_streams:
    i.release()
out.release()
cv2.destroyAllWindows()