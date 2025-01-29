import cv2
import cv2.aruco as aruco
import numpy as np


marker_size = 100

object_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                          [marker_size / 2, marker_size / 2, 0],
                          [marker_size / 2, -marker_size / 2, 0],
                          [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)


class Detector:
    def __init__(self):
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters()

        self.detector = aruco.ArucoDetector(aruco_dict, parameters)

        self.camera_matrix = None
        self.distortion_coefficients = None


    def load_camera(self):
        fs = cv2.FileStorage('camera_parameters.yaml', cv2.FILE_STORAGE_READ)
        self.camera_matrix = fs.getNode('camera_matrix').mat()
        self.distortion_coefficients = fs.getNode('distortion_coefficients').mat()
        fs.release()


    def detect(self, gray_image):
        detections = []

        corners, ids, rejected = self.detector.detectMarkers(gray_image)

        for i, id in enumerate(ids):
            marker_corners = corners[i][0]
            marker_id = ids[0]

            retval, rvec, tvec = cv2.solvePnP(
                    object_points,
                    marker_corners,
                    self.camera_matrix,
                    self.distortion_coefficients
                )

            detections.append({
                'id': marker_id,
                'rvec': rvec,
                'tvec': tvec,
            })

        return detections


def calculate_positions(detections):
    pass

# def load_image():
#     """
#     load an image from source X<> and return it
#     :return: image
#     """
#
# def extract_marker_positions(gray_image):
#     """
#     extract from Image<> marker Number<...>
#     :return:
#     """
#     corners, ids, rejected = detector.detectMarkers(gray)
#
#
# def huhu():
#     """
#     :return:
#     """


if __name__ == "__main__":
    detector = Detector()


    # image = load_image()
    #
    # markers = extract_marker_positions()
    #
    # position = calculate_position()