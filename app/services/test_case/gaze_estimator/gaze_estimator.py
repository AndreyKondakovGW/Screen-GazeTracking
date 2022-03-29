import subprocess
#import tkinter as tk
import cv2
import numpy as np
import sys
import os

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using :0.0')
    os.environ.__setitem__('DISPLAY', ':0.0')


class GazeEstimator:
    def __init__(self,
                 path_gaze_est = "/models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml",
                 path_face_detect = "/models/public/face-detection-retail-0044/FP16/face-detection-retail-0044.xml",
                 path_head_pose_est = "/models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml",
                 path_facial_landmarks = "/models/intel/facial-landmarks-35-adas-0002/FP16/facial-landmarks-35-adas-0002.xml",
                 path_open_closed_eye = "/models/public/open-closed-eye-0001/FP16/open-closed-eye-0001.xml",
                 num_dots = 3,
                 show=False,
                 radius = 50,
                 num_skip_frames=3):

        self.c_outputs = {"left_eye": np.zeros(2),
                          "right_eye": np.zeros(2),
                          "between_eyes": np.zeros(2),
                          "gaze_vector": np.zeros(3),
                          "face_distance": None}

        path_to_dir = "./app/services/test_case/gaze_estimator"
        self.path_gaze_est = path_gaze_est
        self.path_face_detect = path_face_detect
        self.path_head_pose_est = path_head_pose_est
        self.path_facial_landmarks = path_facial_landmarks
        self.path_open_closed_eye = path_open_closed_eye
        self.show = show

        #root = tk.Tk()
        self.screen_width = 1920#root.winfo_screenwidth()
        self.screen_height = 957#root.winfo_screenheight()

        self.dots = np.zeros((num_dots, 2))

        self.popen = None

        self.args = [path_to_dir + "/gaze_estimation_demo",
                "-r",
                "-i", str(0),
                "-m", path_to_dir + self.path_gaze_est,
                "-m_fd", self.path_face_detect,
                "-m_hp", self.path_head_pose_est,
                "-m_lm", self.path_facial_landmarks,
                "-m_es", self.path_open_closed_eye]

        if not self.show:
            self.args.append("-no_show")

        self.heatmap = np.zeros((self.screen_height, self.screen_width))
        self.heatmapshow = np.zeros((self.screen_height, self.screen_width))
        self.r = radius
        self.num_skip_frames = num_skip_frames

    # Runs c++ script
    # Returns the output lines one by one
    def run_c(self):
        os.system("source /opt/intel/openvino_2021/bin/setupvars.sh")
        self.popen = subprocess.Popen(self.args, stdout=subprocess.PIPE, universal_newlines=True)
        for stdout_line in iter(self.popen.stdout.readline, ""):
            yield stdout_line

    # Processes the output from c++ script
    # Writes results to the class fields
    # The function will return 1 when you get all information about another point
    def postprocess_c(self, line):
        print(line)
        if line.strip().startswith("landmark #0:"):
            self.c_outputs["right_eye"] = np.array(eval(line.strip()[13:]))

        elif line.strip().startswith("landmark #2:"):
            self.c_outputs["left_eye"] = np.array(eval(line.strip()[13:]))

            self.c_outputs['between_eyes'] = np.mean([self.c_outputs["left_eye"], self.c_outputs["right_eye"]],
                                                     axis=0)
            self.c_outputs['between_eyes'][0] = self.screen_width - self.c_outputs['between_eyes'][0]

        elif line.strip().startswith("Dist to face:"):
            self.c_outputs["face_distance"] = int(round(float(line.strip()[14:]), 5) * 3793.627)

        elif line.strip().startswith("Gaze vector (x, y, z):"):
            self.c_outputs["gaze_vector"] = np.array(eval(line.strip()[23:]))
            self.c_outputs["gaze_vector"][:-1] *= -1
            return 1

        return 0

    # Computes the location of the gaze on the screen
    # Returns 2d point
    def intersect_point(self):
        m, p, l = self.c_outputs['gaze_vector']
        x, y = self.c_outputs["between_eyes"]
        z = self.c_outputs["face_distance"]

        a = np.array([[p, -m, 0], [0, l, -p], [p, -m, 0], [0, 0, -p]])
        b = np.array([p*x - m*y, l*y - p*z, p*x - m*y, 0])
        dot = np.linalg.lstsq(a, b)[0].round()

        dot[0] = dot[0].clip(0, self.screen_width)
        dot[1] = dot[1].clip(0, self.screen_height)

        return dot[:-1]

    def visualize(self, flag):
        if not flag:

            dot = self.dots.mean(axis=0, dtype=int)
            current_point = np.zeros((self.screen_height, self.screen_width), dtype=np.uint8)
            cv2.circle(current_point, (dot[0], dot[1]), radius=int(self.r * 1.5), color=[128], thickness=-1)
            cv2.circle(current_point, (dot[0], dot[1]), radius=self.r, color=[255], thickness=-1)

            self.heatmap[current_point == 255] += 30
            self.heatmap[current_point == 128] += 15
            self.heatmap[current_point == 0] -= 1

            self.heatmap[self.heatmap < 0] = 0
            self.heatmap[self.heatmap > 255] = 255

            self.heatmapshow = cv2.applyColorMap(self.heatmap.astype('uint8'), cv2.COLORMAP_JET)

        return self.heatmapshow

    def visualize_heatmap(self, flag):
        if not flag:
            total_rows, total_cols = self.heatmap.shape
            X, Y = np.ogrid[:total_rows, :total_cols]

            dot = self.dots.mean(axis=0, dtype=int)
            dist_from_center = np.sqrt((X - dot[1])**2 + (Y-dot[0])**2)

            dist_from_center = 100 - dist_from_center
            dist_from_center[dist_from_center <= 0] = 0

            self.heatmap -= 5
            self.heatmap[self.heatmap < 0] = 0
            self.heatmap = self.heatmap + dist_from_center*5
            

            h_max = np.max(self.heatmap)
            if h_max != 0:
                heatmap_scaled = ((self.heatmap / h_max)*255).astype(int)
            else:
                heatmap_scaled = self.heatmap
            

            self.heatmapshow = cv2.applyColorMap(heatmap_scaled.astype('uint8'), cv2.COLORMAP_JET)

            return self.heatmapshow
    
    def visualize_point(self):
        red = [0, 0, 255]
        dot = self.dots.mean(axis=0, dtype=int)

        src = np.zeros((self.screen_height, self.screen_width, 3), np.uint8)
        cv2.circle(src, (dot[0], dot[1]), radius=50, color=red, thickness=-1)
        return src

    def stop(self):
        cv2.destroyAllWindows()
        self.popen.stdout.close()
        self.popen.terminate()

        return_code = self.popen.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, self.args)

    # Runs executer
    def execute(self):
        try:
            for i, line in enumerate(self.run_c()):
                if self.postprocess_c(line=line):
                    dot = self.intersect_point()
                    self.dots[i % self.dots.shape[0], :] = dot

                    print("screen point:", self.dots.mean(axis=0))
                    self.visualize_heatmap(i % self.num_skip_frames)
                    if cv2.waitKey(10) == 27:
                        break
        finally:
            self.stop()
