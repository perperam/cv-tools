from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np


marker_size = 100

object_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                          [marker_size / 2, marker_size / 2, 0],
                          [marker_size / 2, -marker_size / 2, 0],
                          [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)


def scale_polygon(polygon, scale_factor=0.1):
    # calculate the center of mass
    center = np.mean(polygon, axis=0).astype(np.int32)
    # shift the polygon the origin
    shifted_polygon = polygon - center

    scaled_polygon = shifted_polygon * scale_factor
    # shift the polygon back to its original position
    scaled_polygon += center

    scaled_polygon = scaled_polygon.astype(np.int32)
    return scaled_polygon


def blur_region(image, mask):
    image_blurred = image.copy()
    blurred_region = cv2.GaussianBlur(image, (21, 21), 40)
    image_blurred[mask == 255] = blurred_region[mask == 255]

    return image_blurred


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


    def detect(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Gray", gray_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
                'corners': marker_corners,
                'rvec': rvec,
                'tvec': tvec,
            })

            mask = np.zeros(gray_image.shape[0:2], dtype=np.uint8)

            print(type(gray_image))
            print(marker_corners)
            pts = marker_corners.astype(np.int32)

            # pts = np.array([[357, 336],
            #                 [236, 325],
            #                 [246, 205],
            #                 [366, 215]], dtype=np.int32)

            # manual scaled
            pts = scale_polygon(pts, scale_factor=1.2)

            # marker_corners.astype(np.int32)

            # https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga8c69b68fab5f25e2223b6496aa60dad5

            cv2.fillPoly(mask, [pts], (255,))

            cv2.imshow("Mask", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # cv::INPAINT_NS or cv::INPAINT_TELEA
            inpainted_image = cv2.inpaint(image, mask, inpaintRadius=20, flags=cv2.INPAINT_NS)

            cv2.imshow("Inpainted", inpainted_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            blurred_image = blur_region(inpainted_image, mask)

            cv2.imshow("Blurred", blurred_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return detections


if __name__ == "__main__":
    detector = Detector()
    detector.load_camera()

    image_path = Path("cap_00.jpg")

    image = cv2.imread(str(image_path))

    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    detector.detect(image)