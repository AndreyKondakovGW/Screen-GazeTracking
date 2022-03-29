import numpy as np

def convert_gaze_vector_to_screen(gaze_vector, between_eyes, face_distance, screen_width, screen_height):
    num_dots=10
    dots = np.zeros((num_dots, 2))
    gaze_vector[:-1] *= -1
    m, p, l = gaze_vector[0]

    x, y = between_eyes

    z = face_distance * 3793.627

    a = np.array([[p, -m, 0], [0, l, -p], [p, -m, 0], [0, 0, -p]])
    b = np.array([p*x - m*y, l*y - p*z, p*x - m*y, 0])
    dot = np.linalg.lstsq(a, b)[0].round()

    # dot[1] -= 600
    # dot[0] -= 150

    dot[0] = dot[0].clip(0, screen_width)
    dot[1] = dot[1].clip(0, screen_height)

    return dot[:-1]

