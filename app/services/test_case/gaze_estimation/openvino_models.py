import cv2
import numpy as np

from openvino.inference_engine import IECore
from helpers import rotate_image, shape_to_np

face_detection_path = "./models/public/face-detection-retail-0044/FP32"
facial_landmarks_path = "./models/intel/facial-landmarks-35-adas-0002/FP32"
head_pose_estimation_path = "./models/intel/head-pose-estimation-adas-0001/FP32"
gaze_estimation_path = "./models/intel/gaze-estimation-adas-0002/FP32"
open_closed_eye_path = "./models/public/open-closed-eye-0001/FP32"

cv_cascade_path = "./models/cvEyesDetect/haarcascade_eye.xml"
dlib_model_path = "./models/cvEyesDetect/shape_predictor_68_face_landmarks.dat"

def create_openvino__nets():
    print('Creating Inference Engine')
    ie = IECore()

    print('Creating FD Net')
    face_det_net = ie.read_network(model =face_detection_path +"/face-detection-retail-0044.xml",
                                  weights=face_detection_path +"/face-detection-retail-0044.bin")

    face_det_exec_net = ie.load_network(face_det_net, "CPU")

    face_output_blob = next(iter(face_det_exec_net.outputs))
    face_input_blob = next(iter(face_det_exec_net.input_info))

    face_net = {"net":face_det_net ,"exec_net": face_det_exec_net, "input_blob": face_input_blob, "output_blob": face_output_blob}

    
    print('Creating FL Net')
    facial_landmarks_net = ie.read_network(model =facial_landmarks_path +"/facial-landmarks-35-adas-0002.xml",
                                  weights=facial_landmarks_path +"/facial-landmarks-35-adas-0002.bin")

    facial_landmarks_exec_net = ie.load_network(facial_landmarks_net, "CPU")

    landmarks_output_blob = next(iter(facial_landmarks_exec_net.outputs))
    landmarks_input_blob = next(iter(facial_landmarks_exec_net.input_info))

    landmarks_net = {"net":facial_landmarks_net ,"exec_net": facial_landmarks_exec_net, "input_blob": landmarks_input_blob, "output_blob": landmarks_output_blob}
    

    print('Creating HP Net')
    head_pos_net = ie.read_network(model =head_pose_estimation_path  +"/head-pose-estimation-adas-0001.xml",
                                  weights=head_pose_estimation_path  +"/head-pose-estimation-adas-0001.bin")

    head_pos_exec_net = ie.load_network(head_pos_net, "CPU")

    head_output_blob = next(iter(head_pos_exec_net.outputs))
    head_input_blob = next(iter(head_pos_exec_net.input_info))

    head_net = {"net":head_pos_net ,"exec_net": head_pos_exec_net, "input_blob": head_input_blob, "output_blob": head_output_blob}

    print('Creating GE Net')
    gaze_estim_net = ie.read_network(model =gaze_estimation_path  +"/gaze-estimation-adas-0002.xml",
                                  weights=gaze_estimation_path  +"/gaze-estimation-adas-0002.bin")

    gaze_estim_exec_net = ie.load_network(gaze_estim_net, "CPU")

    gaze_output_blob = next(iter(gaze_estim_exec_net.outputs))
    gaze_input_blob = next(iter(gaze_estim_exec_net.input_info))

    gaze_net = {"net":gaze_estim_net ,"exec_net": gaze_estim_exec_net, "input_blob": gaze_input_blob, "output_blob": gaze_output_blob}

    return face_net,landmarks_net,head_net,gaze_net

def create_cv_nets():
    eye_cascade = cv2.CascadeClassifier(cv_cascade_path)

    return eye_cascade

def detect_faces(model, img, strategy="biggest"):
    image = img.copy()
    _, _, net_h, net_w = model["net"].input_info[model["input_blob"]].input_data.shape
    if image.shape[:-1] != (net_h, net_w):
        image = cv2.resize(image, (net_w, net_h))

    # Change data layout from HWC to CHW
    image = image.transpose((2, 0, 1))
    # Add N dimension to transform to NCHW
    image = np.expand_dims(image, axis=0)

    res = model["exec_net"].infer(inputs={model["input_blob"]: image})

    h, w, _ = img.shape

    if len(model["net"].outputs) == 1:
        res = res[model["output_blob"]]
        # Change a shape of a numpy.ndarray with results ([1, 1, N, 7]) to get another one ([N, 7]),
        # where N is the number of detected bounding boxes
        detections = res.reshape(-1, 7)
    else:
        detections = res['boxes']

    faces = []
    main_face = []
    for i, detection in enumerate(detections):
        _, _, confidence, xmin, ymin, xmax, ymax = detection

        if confidence > 0.5:
            xmin = int(xmin * w)
            ymin = int(ymin * h)
            xmax = int(xmax * w)
            ymax = int(ymax * h)
            faces.append((xmin, ymin, xmax, ymax))

        #Сохраняем главное лицо (либо самое большое либо самое чётокое)
        if strategy == "biggest":
            if i == 0:
                main_face.append((xmin, ymin, xmax, ymax))
                max_face_size = abs(xmax - xmin) * abs(ymax - ymin)
            else:
                if abs(xmax - xmin) * abs(ymax - ymin) > max_face_size:
                    main_face[0] = (xmin, ymin, xmax, ymax)
                    max_face_size = abs(xmax - xmin) * abs(ymax - ymin)
        else:
            if i == 0:
                main_face.append((xmin, ymin, xmax, ymax))
                max_face_conf = confidence
            else:
                if confidence > max_face_conf:
                    main_face[0] = (xmin, ymin, xmax, ymax)
                    max_face_conf = confidence

    return main_face

def get_face_landmark(model, img):
    image = img.copy()
    _, _, net_h, net_w = model["net"].input_info[model["input_blob"]].input_data.shape
    if image.shape[:-1] != (net_h, net_w):
        image = cv2.resize(image, (net_w, net_h))
    
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)

    res = model["exec_net"].infer(inputs={model["input_blob"]: image})

    h, w, _ = img.shape
    if len(model["net"].outputs) == 1:
        res = res[model["output_blob"]]
        detection_points = res.reshape(-1, 70)
        
    else:
        detection_points = res['align_fc3']  
    
    landmark_points = []
    for i in range(0,35):
        xi = int(detection_points[0][i*2] * w)
        yi = int(detection_points[0][i*2+1] * h)
        landmark_points.append((xi,yi))
    
    return landmark_points

def detect_eyes(model, img):
    image = img.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    eyes = model.detectMultiScale(gray, minNeighbors=5)
    eyes_pos = []
    for (ex,ey,ew,eh) in eyes:
        eyes_pos.append((ex,ey, ex+ew,ey+eh)) #Cелать проверку что площадь глаза не нулевая
    
    return eyes_pos

def get_head_angle(model, img):
    image = img.copy()
    _, _, net_h, net_w = model["net"].input_info[model["input_blob"]].input_data.shape
    if image.shape[:-1] != (net_h, net_w):
        image = cv2.resize(image, (net_w, net_h))

    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)

    res = model["exec_net"].infer(inputs={model["input_blob"]: image})

    head_angle = (res["angle_y_fc"][0], res["angle_p_fc"][0], res["angle_r_fc"][0])
    return head_angle

def get_gaze_vector(model, img, head_angle, eyes_pos):
    image = img.copy()
    right_eye_pos = eyes_pos[0]
    left_eye_pos = eyes_pos[1]
    if eyes_pos[0][0] < eyes_pos[1][0]:
        right_eye_pos = eyes_pos[1]
        left_eye_pos = eyes_pos[0]
    right_eye_croped = image[right_eye_pos[1]:right_eye_pos[3], right_eye_pos[0]:right_eye_pos[2]]
    left_eye_croped = image[left_eye_pos[1]:left_eye_pos[3], left_eye_pos[0]:left_eye_pos[2]]

    yaw, pitch, roll = head_angle

    right_eye_croped = rotate_image(right_eye_croped, roll)
    left_eye_croped = rotate_image(left_eye_croped, roll)
    #cv2.imshow('RE1', right_eye_croped)
    #cv2.imshow('RE2', left_eye_croped)

    _, _, re_net_h, re_net_w = model["net"].input_info["right_eye_image"].input_data.shape
    _, _, le_net_h, le_net_w = model["net"].input_info["left_eye_image"].input_data.shape

    if right_eye_croped.shape[:-1] != (re_net_h, re_net_w):
        right_eye_croped = cv2.resize(right_eye_croped, (re_net_h, re_net_w))
    if left_eye_croped.shape[:-1] != (le_net_h, le_net_w):
        left_eye_croped = cv2.resize(left_eye_croped, (le_net_h, le_net_w))
    
    # Change data layout from HWC to CHW
    right_eye_croped = right_eye_croped.transpose((2, 0, 1))
    left_eye_croped = left_eye_croped.transpose((2, 0, 1))
    # Add N dimension to transform to NCHW
    right_eye_croped = np.expand_dims(right_eye_croped, axis=0)
    left_eye_croped = np.expand_dims(left_eye_croped, axis=0)

    angles = np.array([yaw, pitch, roll])
    angles = angles.reshape((1,3))
    res = model["exec_net"].infer(inputs={"right_eye_image": right_eye_croped, "left_eye_image": left_eye_croped,"head_pose_angles": angles})
    gaze_vector = res["gaze_vector"] / cv2.norm(res["gaze_vector"])
    return gaze_vector
    
        
