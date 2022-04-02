import cv2
from math import sqrt
import numpy as np

def get_eyes_cords(landmark_points):
    x0, y0 = landmark_points[0]
    x1, y1 = landmark_points[1]

    x2, y2 = landmark_points[2]
    x3, y3 = landmark_points[3]

    right_eye_width = abs(x0 - x1)
    right_eye_heigh = abs(y0 - y1)
    r_mid_x, r_mid_y = (x0 + x1) // 2, (y0 + y1) // 2

    right_eye = (r_mid_x - right_eye_width//2, r_mid_y - right_eye_width//2, r_mid_x + right_eye_width//2, r_mid_y + right_eye_width//2)

    left_eye_width = abs(x2 - x3)
    left_eye_heigh = abs(y2 - y3)
    l_mid_x, l_mid_y = (x3 + x2) // 2, (y2 + y3) // 2

    left_eye = (l_mid_x - left_eye_width//2, l_mid_y - left_eye_width//2, l_mid_x + left_eye_width//2, l_mid_y + left_eye_width//2)

    return right_eye, left_eye


def get_eyes_cords_cv(landmark_points, eyes_positions):
    r_e_p = []
    l_e_p = []
    lp0 = landmark_points[0]
    lp1 = landmark_points[1]
    lp2 = landmark_points[2]
    lp3 = landmark_points[3]
    for eye_pos in eyes_positions:
        e_p_x1, e_p_y1 = eye_pos[0],eye_pos[1]
        e_p_x2, e_p_y2 = eye_pos[2],eye_pos[3] 
        if e_p_x1<=lp0[0] and e_p_y1<=lp0[1] and e_p_x2>=lp1[0] and e_p_y2>=lp1[1]:
            if e_p_x2<=lp2[0]:
                r_e_p = [e_p_x1, e_p_y1, e_p_x2, e_p_y2]
        
        if e_p_x1<=lp2[0] and e_p_y1<=lp2[1] and e_p_x2>=lp3[0] and e_p_y2>=lp3[1]:
            if e_p_x1>=lp1[0]:
                l_e_p = [e_p_x1, e_p_y1, e_p_x2, e_p_y2]
    return  r_e_p, l_e_p

def get_eyes_cords_from_landmarks(landmark_points):
    lp0 = landmark_points[0]
    lp1 = landmark_points[1]
    lp2 = landmark_points[2]
    lp3 = landmark_points[3]

    l_e = get_eyes_box(lp0, lp1)
    r_e = get_eyes_box(lp2, lp3)

    return  r_e, l_e



def get_eyes_box(p1, p2, scale = 1.8):
    width = abs(p1[0] - p2[0])
    height = abs(p1[1] - p2[1])
    size = sqrt(height**2 + width**2)

    eye_width = int(size * scale)
    eye_height = eye_width

    mid_pointX = (p1[0] + p2[0]) // 2
    mid_pointY = (p1[1] + p2[1]) // 2

    x1,x2 = mid_pointX - (eye_width // 2), mid_pointX + (eye_width // 2)
    y1,y2 = mid_pointY - (eye_height // 2), mid_pointY + (eye_height // 2)

    return [x1, y1, x2, y2]


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, int(angle), 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
