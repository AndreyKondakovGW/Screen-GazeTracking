import pyrealsense2 as rs
import numpy as np
import cv2

from helpers import *
from models import *

class GazeEstimator:
    def __init__(self,
    num_dots=5,
    use_intel_camera = False,
    show_face = False,
    show_eyes = False,
    show_landmarks = False,
    show_gaze_point = True,
    path_background_photo = "background_image.jpg"):
        self.use_intel_camera = use_intel_camera
        self.show_face = show_face
        self.show_eyes = show_eyes
        self.show_landmarks = show_landmarks
        self.show_gaze_point = show_gaze_point

        #Сколько последних кадров используется для расчёта вектора взгляда
        self.num_dots = num_dots
        #Массив последних точек взгляда
        self.dots = np.zeros((num_dots, 2))

        self.win_width = 1920
        self.win_height = 1080

        self.heatmap = np.zeros((self.win_height, self.win_width)) 

        self.face_net, self.landmarks_net, self.head_pos_net, self.gaze_net = create_openvino__nets()
        self.eyes_net = create_cv_nets()

        bg_src = cv2.imread(path_background_photo)
        self.bg = cv2.resize(bg_src, (self.win_width, self.win_height))

        #Инициализируем камеру
        if self.use_intel_camera:
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
            pipeline_profile = self.config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            device_product_line = str(device.get_info(rs.camera_info.product_line))

            found_rgb = False
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True
                    break
            if not found_rgb:
                print("The demo requires Depth camera with Color sensor")
                self.use_intel_camera = False
            
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

            if device_product_line == 'L500':
                self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
            else:
                self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) 
        else:
            self.cap = cv2.VideoCapture(0)    

    def release_camera(self):
        cv2.destroyAllWindows()
        if self.use_intel_camera:
            self.pipeline.stop()
        else:
            self.cap.release()

    def detect_face(self, color_frame, view_image):
        faces = detect_faces(self.face_net, color_frame)
        if len(faces) > 0:
            xmin, ymin, xmax, ymax = faces[0]
            cx, cy = (xmax + xmin) // 2, (ymax + ymin) // 2

            if (self.show_face):
                cv2.rectangle(view_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

            crop_face = color_frame[ymin:ymax, xmin:xmax]
            face_pos = (xmin, ymin, xmax, ymax)
            return (True, crop_face, cx, cy, face_pos)
        else:
            return (False, 0, 0, 0, 0)

    def detect_eyes(self, crop_face, face_pos, color_frame, view_image):
        xmin, ymin, xmax, ymax = face_pos
        landmark_points = get_face_landmark(self.landmarks_net, crop_face)

        if self.show_landmarks:
            for (x,y) in landmark_points:
                cv2.circle(view_image, (xmin + x,ymin + y), 1, (0, 255, 0), 1)
        eyes_positions = detect_eyes(self.eyes_net, crop_face)

        #Если глаза найденны находим их центр и ищем вектор взгляда
        if len(landmark_points) != 0 and len(eyes_positions) != 0:
            #r_e_p, l_e_p = get_eyes_cords_cv(landmark_points, eyes_positions)
            r_e_p, l_e_p = get_eyes_cords_from_landmarks(landmark_points)
                            
            if len(r_e_p) != 0 and len(l_e_p) != 0:
                r_mp = ((r_e_p[0] + r_e_p[2]) // 2, (r_e_p[1] + r_e_p[3]) // 2)
                l_mp = ((l_e_p[0] + l_e_p[2]) // 2, (l_e_p[1] + l_e_p[3]) // 2)
                return True, eyes_positions, r_e_p, l_e_p
            else:
                return False, 0, 0, 0
        else:
            return False, 0, 0, 0

    def detect_gaze_vector(self, crop_face, face_pos, head_angle, r_e_p, l_e_p, view_image):
        xmin, ymin, xmax, ymax = face_pos
        gaze_vector = get_gaze_vector(self.gaze_net, crop_face, head_angle, [r_e_p, l_e_p])
        print(f"Gaze vector {gaze_vector[0]}")

        between_eyss_point = (np.mean(np.array([r_e_p[0], r_e_p[2], l_e_p[0], l_e_p[2]])),np.mean(np.array([r_e_p[1], r_e_p[3], l_e_p[1], l_e_p[3]])))
        if self.show_eyes:
            cv2.rectangle(view_image, (xmin +r_e_p[0], ymin +r_e_p[1]), (xmin +r_e_p[2], ymin +r_e_p[3]), (0, 0, 255), 3)
            cv2.rectangle(view_image, (xmin +l_e_p[0], ymin +l_e_p[1]), (xmin +l_e_p[2], ymin +l_e_p[3]), (0, 0, 255), 3)
            cv2.circle(view_image, (xmin + int(between_eyss_point[0]),ymin + int(between_eyss_point[1])), 1, (0, 255, 0), 1)

        return gaze_vector[0], between_eyss_point

    #Функция детектет на картинке лицо и определяет по нему вектор взгляда (нужна для тестов)
    def get_info_by_image(self, path):
        color_frame = cv2.imread(path)
        view_image = color_frame.copy()
        face_detected, crop_face, cx, cy, face_pos = self.detect_face(color_frame, view_image)
        if (face_detected):
            eyes_detected, eyes_positions, r_e_p, l_e_p  = self.detect_eyes(crop_face, face_pos, color_frame, view_image)
            if (eyes_detected):
                head_angle = get_head_angle(self.head_pos_net, crop_face)
                gaze_vector, between_eyss_point = self.detect_gaze_vector(crop_face, face_pos, head_angle, r_e_p, l_e_p, view_image)
            else:
                print('Eyes not detected')
        else:
            print('Face not detected')

        cv2.namedWindow('Image')
        cv2.imshow('Image', view_image)
        cv2.waitKey(0)


    def gen_frames(self):
        if self.use_intel_camera:
            # запуск камеры
            self.pipeline.start(self.config)
        try:
            frame_counter = -1
            ret = True
            while True:
                frame_counter = (frame_counter + 1) % self.num_dots
                if self.use_intel_camera:
                    frames = self.pipeline.wait_for_frames()
                    color_frame, depth_frame =  parse_frames(frames)
                else:
                    ret, color_frame = self.cap.read()
                
                if ret:
                    view_image = color_frame.copy()
                else:
                    continue
                
                #Определение лиц на экране
                face_detected, crop_face, cx, cy, face_pos = self.detect_face(color_frame, view_image)

                if (face_detected):
                    #Оперделим расстояние до лица
                    if self.use_intel_camera:
                        dist_to_face = frames.get_depth_frame().get_distance(cx, cy)
                    else:
                        dist_to_face = 0.4

                    #Обрезаем квадрат лица и ищем на нём глаза
                    #Для поиска глаз сначала при помощи опенвино сети находим все точки глаз
                    #Потом с их помощью сверяем результат OpenCV сети для поиска глаз
                    eyes_detected, eyes_positions, r_e_p, l_e_p  = self.detect_eyes(crop_face, face_pos, color_frame, view_image)

                    if (eyes_detected):
                        #Определяем угол поворота головы
                        head_angle = get_head_angle(self.head_pos_net, crop_face)

                        gaze_vector, between_eyss_point = self.detect_gaze_vector(crop_face, face_pos, head_angle, r_e_p, l_e_p, view_image)

                        #Конвертируем вектор взгляда в точку на экране
                        gaze_point = convert_gaze_vector_to_screen(gaze_vector, between_eyss_point, dist_to_face, self.win_width, self.win_height)
                        self.dots[frame_counter, :] = gaze_point

                        #Визуализируем полученные картинки
                        if self.show_gaze_point:
                            self.heatmapshow = visualize(frame_counter % 3, self.dots, self.win_height, self.win_width, self.heatmap, self.bg)
                            #print(self.heatmapshow)
                            cv2.namedWindow('Gaze Point', cv2.WINDOW_FULLSCREEN)
                            cv2.imshow('Gaze Point', self.heatmapshow)

                if self.show_face or self.show_landmarks:
                    cv2.namedWindow('RealSense face', cv2.WINDOW_AUTOSIZE) 
                    cv2.imshow('RealSense face', view_image)
                
                if cv2.waitKey(1) == 27:
                    cv2.destroyAllWindows()
                    break
        finally:
            self.release_camera()



gaze_estim = GazeEstimator(show_gaze_point = True, use_intel_camera = True, show_face = True, show_eyes = True)
gaze_estim.get_info_by_image('./face.jpg')
#gaze_estim.gen_frames() 