import pyrealsense2 as rs
import numpy as np
import cv2
import tkinter as tk

from openvino_models import create_openvino__nets, detect_faces, get_face_landmark, get_head_angle, create_cv_nets, detect_eyes, get_gaze_vector
from helpers import get_eyes_cords, get_eyes_cords_cv
from gaze_vector_converter import convert_gaze_vector_to_screen
from realsence_helper import parse_frames

def main():
    
    USE_INTEL_CAMERA = False

    #Определение размера экрана (Выпилить для запуска в докере)
    root = tk.Tk()
    WIN_WIDTH  = root.winfo_screenwidth()
    WIN_HEIGHT  = root.winfo_screenheight()
    print(f"{WIN_WIDTH}, {WIN_HEIGHT}")
    
    #WIN_WIDTH = 1920
    #WIN_HEIGHT = 957

    SHOW_FACE = True
    SHOW_EYES = True
    SHOW_LANDMARKS = True
    SHOW_GAZE_POINT = True
    DEVICE = "CPU"


    face_net, landmarks_net, head_pos_net,gaze_net = create_openvino__nets()
    eyes_net = create_cv_nets()

    #Настройка Intel камеры
    if USE_INTEL_CAMERA:
        pipeline = rs.pipeline()
        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            USE_INTEL_CAMERA = False

    if USE_INTEL_CAMERA:
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # запуск камеры
        pipeline.start(config)
    else:
        cap = cv2.VideoCapture(-1)

    try:
        while True:
            ret = True
            #Получение кадров либо с Intel камеры либо с Вебки
            if USE_INTEL_CAMERA:
                frames = pipeline.wait_for_frames()
                color_frame,depth_frame =  parse_frames(frames)
            else:
                ret, color_frame = cap.read()
                print(ret)

            if (ret):
                print(ret)
                print(color_frame)
                view_image = color_frame.copy()
            else:
                continue
            
            #Определение лиц на экране
            faces = detect_faces(face_net, color_frame)
            for i, (xmin, ymin, xmax, ymax) in enumerate(faces):
                cx, cy = (xmax + xmin) // 2, (ymax + ymin) // 2

                if USE_INTEL_CAMERA:
                    dist_to_face = frames.get_depth_frame().get_distance(cx, cy)
                else:
                    dist_to_face = 0.4

                if (SHOW_FACE):
                    cv2.rectangle(view_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                    cv2.putText(view_image, "dist to centre = " + str(dist_to_face), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

                #Обрезаем квадрат лица и ищем на нём глаза
                #Для поиска глаз сначала при помощи опенвино сети находим все точки глаз
                #Потом с их помощью сверяем результат OpenCV сети для поиска глаз
                crop_face = color_frame[ymin:ymax, xmin:xmax]
                if xmax>=0 and xmin>=0 and ymax>=0 and ymin>=0:
                    landmark_points = get_face_landmark(landmarks_net, crop_face)
                    if SHOW_LANDMARKS:
                        for (x,y) in landmark_points:
                            cv2.circle(view_image, (xmin + x,ymin + y), 1, (0, 255, 0), 1)
                    eyes_positions = detect_eyes(eyes_net, crop_face)

                    #Определяем угол поворота головы
                    head_angle = get_head_angle(head_pos_net, crop_face)

                    #Если глаза найденны находим их центр и ищем вектор взгляда
                    if len(landmark_points) != 0 and len(eyes_positions) != 0:
                        r_e_p, l_e_p = get_eyes_cords_cv(landmark_points, eyes_positions)
                        
                        if len(r_e_p) != 0 and len(l_e_p) != 0:
                            r_mp = ((r_e_p[0] + r_e_p[2]) // 2, (r_e_p[1] + r_e_p[3]) // 2)
                            l_mp = ((l_e_p[0] + l_e_p[2]) // 2, (l_e_p[1] + l_e_p[3]) // 2)
                            
                            gaze_vector = get_gaze_vector(gaze_net, crop_face, head_angle, [r_e_p, l_e_p])
                            print(f" Gaze vector {gaze_vector}")

                            between_eyss_point = (np.mean(np.array([r_e_p[0], r_e_p[2], l_e_p[0], l_e_p[2]])),np.mean(np.array([r_e_p[1], r_e_p[3], l_e_p[1], l_e_p[3]])))
                            if SHOW_EYES:
                                cv2.rectangle(view_image, (xmin +r_e_p[0], ymin +r_e_p[1]), (xmin +r_e_p[2], ymin +r_e_p[3]), (0, 0, 255), 3)
                                cv2.rectangle(view_image, (xmin +l_e_p[0], ymin +l_e_p[1]), (xmin +l_e_p[2], ymin +l_e_p[3]), (0, 0, 255), 3)
                                cv2.circle(view_image, (xmin + int(between_eyss_point[0]),ymin + int(between_eyss_point[1])), 1, (0, 255, 0), 1)

                            gaze_point = convert_gaze_vector_to_screen(gaze_vector, between_eyss_point, dist_to_face, WIN_WIDTH, WIN_HEIGHT)
                            src = np.zeros((WIN_WIDTH,WIN_HEIGHT, 3), np.uint8)
                            cv2.circle(src, (int(gaze_point[0]), int(gaze_point[1])), radius=20, color=(0,0,255), thickness=-1)
                            if SHOW_GAZE_POINT:
                                cv2.namedWindow('Gaze Point', cv2.WINDOW_FULLSCREEN)
                                cv2.imshow('Gaze Point', src)

            
            if SHOW_FACE or SHOW_LANDMARKS: 
                cv2.namedWindow('RealSense face', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense face', view_image)

            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                break

    finally:
        cv2.destroyAllWindows()
        if USE_INTEL_CAMERA:
            pipeline.stop()
        else:
            cap.release()

main()