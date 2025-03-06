import csv
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation
import scipy


marker_size = 30

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


    def detect(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Gray", gray_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        detections = []

        corners, ids, rejected = self.detector.detectMarkers(gray_image)


        for i, id in enumerate(ids):
            marker_corners = corners[i][0]
            marker_id = ids[i]

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


        return detections


class Cleaner:
    def clean(self, image: np.ndarray, detections: list[dict], verbose: bool=False) -> np.ndarray:
        for detection in detections:
            mask = np.zeros(image.shape[0:2], dtype=np.uint8)

            corners = detection['corners'].astype(np.int32)

            # manual scaled
            corners = self.scale_polygon(corners, scale_factor=1.2)

            # marker_corners.astype(np.int32)

            # https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga8c69b68fab5f25e2223b6496aa60dad5

            cv2.fillPoly(mask, [corners], (255,))

            if verbose: self.show_image("Mask", mask)

            # cv::INPAINT_NS or cv::INPAINT_TELEA
            inpainted_image = cv2.inpaint(image, mask, inpaintRadius=20, flags=cv2.INPAINT_NS)

            if verbose: self.show_image("Inpainted", inpainted_image)

            blurred_image = self.blur_region(inpainted_image, mask)

            if verbose: self.show_image("Blurred", blurred_image)

            image = blurred_image

        return image


    def show_image(self, title: str, image: np.ndarray):
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def scale_polygon(self, polygon, scale_factor=0.1):
        # calculate the center of mass
        center = np.mean(polygon, axis=0).astype(np.int32)
        # shift the polygon the origin
        shifted_polygon = polygon - center

        scaled_polygon = shifted_polygon * scale_factor
        # shift the polygon back to its original position
        scaled_polygon += center

        scaled_polygon = scaled_polygon.astype(np.int32)
        return scaled_polygon


    def blur_region(self, image, mask):
        image_blurred = image.copy()
        blurred_region = cv2.GaussianBlur(image, (21, 21), 40)
        image_blurred[mask == 255] = blurred_region[mask == 255]

        return image_blurred



class Calculator:
    def calculate(self, detections: list[dict]):
        rotations: list[np.ndarray] = []
        translations: list[np.ndarray] = []

        for detection in detections:
            rotations.append(self.change_rotation(detection['rvec']))
            translations.append(np.array(detection['tvec'].flatten()))

        rotations: np.ndarray = np.array(rotations)
        rotation_means: list[float] = []
        for rotation in rotations.T:
            rotation_mean = scipy.stats.circmean(rotation)

            rotation_means.append(rotation_mean)

        rotation_means: np.ndarray = np.array(rotation_means)


        translations: np.ndarray = np.array(translations)
        translation_means: list[float] = []
        for translation in translations.T:
            translation_mean = np.mean(translation)

            translation_means.append(translation_mean)

        translation_means: np.ndarray = np.array(translation_means)

        # the 3d rotation and the translation as mean over each axis of detection
        return rotation_means, translation_means


    def change_rotation(self, rvec):
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        rot = Rotation.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)
        return rot


def process_images(images_path:Path, csv_path:Path, export_path: None|Path=None):
    detector = Detector()
    detector.load_camera()

    cleaner = Cleaner()
    calculator = Calculator()


    if export_path:
        for camera_side in ['left', 'right']:
            (export_path / camera_side).mkdir(exist_ok=True, parents=True)


    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')

        writer.writerow(['tx', 'ty', 'tz', 'rx', 'ry', 'rz'])


        left_images: Path = images_path / 'left'
        right_images: Path = images_path / 'right'

        for images in zip(left_images.glob('*'), right_images.glob('*')):
            left_image_path = images[0]
            right_image_path = images[1]

            left_image = cv2.imread(str(left_image_path))
            right_image = cv2.imread(str(right_image_path))

            left_detection = detector.detect(left_image)
            right_detection = detector.detect(right_image)

            if export_path:
                left_cleaned_image = cleaner.clean(left_image, left_detection)
                right_cleaned_image = cleaner.clean(right_image, right_detection)

                left_export_path = export_path / 'left'
                right_export_path = export_path / 'right'

                cv2.imwrite(str(left_export_path), left_cleaned_image)
                cv2.imwrite(str(right_export_path), right_cleaned_image)

            left_rotation, left_translation = calculator.calculate(left_detection)
            right_rotation, right_translation = calculator.calculate(right_detection)


            rotations = np.array([left_rotation, right_rotation])
            translations = np.array([left_translation, right_translation])


            rotation_mean = scipy.stats.circmean(rotations, axis=0)  # can return array even if not documented
            translation_mean = np.mean(translations, axis=0)


            row = list(rotation_mean) + list(translation_mean)
            writer.writerow(row)





if __name__ == "__main__":
    process_images(Path.cwd(), Path.cwd() / 'label.csv')