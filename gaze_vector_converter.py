import numpy as np
import cv2

def convert_gaze_vector_to_screen(gaze_vector, between_eyes, face_distance, screen_width, screen_height):
    eps = 1e-9
    norm = np.linalg.norm(gaze_vector)
    gaze_vector /= (1 if norm <= eps else norm)
    #gaze_vector = rotate(gaze_vector, between_eyes, screen_width)
    gaze_vector[:-1] *= -1
    m, p, l = gaze_vector

    x, y = between_eyes

    z = face_distance * 3793.627

    a = np.array([[p, -m, 0], [0, l, -p], [p, -m, 0], [0, 0, -p]])
    b = np.array([p*x - m*y, l*y - p*z, p*x - m*y, 0])
    dot = np.linalg.lstsq(a, b)[0].round()


    dot[0] = int(dot[0].clip(0, screen_width))
    dot[1] = int(dot[1].clip(0, screen_height))

    return dot[:-1]

def rotate(gaze_vector, between_eyes, screen_width):
    cx, cy, cz = screen_width/2, 0, 0
    fx, fy = between_eyes
    fz = gaze_vector

    cam_p = np.array([-(cx - fx), -(cy - fy), -(cz - fz)])

    yz_proj_p = np.array([0, -(cy - fy), -(cz - fz)])
    xz_proj_p = np.array([0, 0, -(cz - fz)])

    alpha = np.arccos(cam_p.dot(yz_proj_p) / (np.linalg.norm(cam_p) * np.linalg.norm(yz_proj_p)))
    beta = np.arccos(yz_proj_p.dot(xz_proj_p) / (np.linalg.norm(yz_proj_p) * np.linalg.norm(xz_proj_p)))

    rotateMatrX = lambda phi: np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    rotateMatrY = lambda phi: np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]])

    rotateY = rotateMatrY(alpha * np.sign(gaze_vector[0]) * -1).dot(gaze_vector)
    rotateYX = rotateMatrX(beta * np.sign(gaze_vector[1])).dot(rotateY)

    return rotateYX

def visualize(flag, dots, screen_height, screen_width, heatmap, background):
    dot = dots.mean(axis=0, dtype=int)
    print(dot)

    mask = np.ogrid[:screen_height, :screen_width]
    dist = np.sqrt((mask[0] - dot[1])**2 + (mask[1] - dot[0])**2)
    dist = 50 - dist
    dist[dist < 0] = 0

    heatmap += 5*dist
    heatmap *= 0.99

    scale_hm = heatmap
    scale_const = np.max(heatmap)

    if scale_const:
        scale_hm = heatmap / scale_const * 255

    heatmapshow = cv2.applyColorMap(scale_hm.astype('uint8'), cv2.COLORMAP_JET)
    bg_with_heatmap = cv2.addWeighted(heatmapshow, 0.8, background, 0.2, 0)

    return bg_with_heatmap


