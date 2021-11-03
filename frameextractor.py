# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:52:08 2021

@author: chakati
"""
# code to get the key frame from the video and save it as a png file.

import cv2
import os


# videoPath : path of the video file
# frames_path: path of the directory to which the frames are saved
# count: to assign the video order to the frame.
def frame_extractor(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frame_no = int(total_frames / 2)
    cap.set(1, frame_no)
    ret, frame = cap.read()
    return frame
