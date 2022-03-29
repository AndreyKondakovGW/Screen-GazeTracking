import imp
import cv2
import multiprocessing as mp

from app.services.test_case.gaze_estimator import GazeEstimator

class TestCase:
    def __init__(self):
        #self.camera = cv2.VideoCapture(0)
        self.estimator = GazeEstimator()

    def gen_frames(self, q_output, process_status):
        try:
            for i, line in enumerate(self.estimator.run_c()):
                if not process_status.value:
                    self.estimator.stop()
                    break
                if self.postprocess_c(line=line):
                    dot = self.estimator.intersect_point()
                    self.estimator.dots[i % self.dots.shape[0], :] = dot

                    heatmap = self.estimator.visualize_heatmap(i % self.estimator.num_skip_frames)
                    _, frame = cv2.imencode('.jpg', heatmap)
                    q_output.put_nowait(frame)
        finally:
            self.estimator.stop()
                


def case_start(q_output: mp.Queue, process_status: mp.Value):
    case = TestCase()
    case.gen_frames(q_output, process_status)
