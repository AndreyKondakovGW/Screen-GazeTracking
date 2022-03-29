#!/bin/bash

face_detection_path=./models/public/face-detection-retail-0044/FP16/face-detection-retail-0044.xml
head_pose_estimation_path=./models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml
facial_landmarks_path=./models/intel/facial-landmarks-35-adas-0002/FP16/facial-landmarks-35-adas-0002.xml
gaze_estimation_path=./models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml
open_closed_eye_path=./models/public/open-closed-eye-0001/FP16/open-closed-eye-0001.xml

./gaze_estimation_demo -r -i -1 -m $gaze_estimation_path -m_fd $face_detection_path -m_hp $head_pose_estimation_path -m_lm $facial_landmarks_path -m_es $open_closed_eye_path