"""
Author: https://github.com/perperam

OpenCV Camera Coordinate System
https://docs.opencv.org/3.4/d5/d1f/calib3d_solvePnP.html

How to calculate the mean of quaternions using skipy
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.concatenate.html#scipy.spatial.transform.Rotation.concatenate
"""

import argparse
from pathlib import Path
import json

import numpy as np
import cv2
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the intrinsic camera parameters
    :param path: the directory path where the calibration yaml files are stored
    :return: the camera matrix and the distortion coefficients
    """
    fs = cv2.FileStorage(str(path / 'camera_parameters.yaml'), cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode('camera_matrix').mat()
    distortion_coefficients = fs.getNode('distortion_coefficients').mat()
    fs.release()

    return camera_matrix, distortion_coefficients


def gen_homogenous(rot_m, trans_vec) -> np.ndarray:
    """
    Generates a homogenous matrix out of the rotation matrix and translation vector.
    :param rot_m: rotation matrix
    :param trans_vec: translation vector
    :return: the homogenous matrix
    """
    T = np.eye(4)
    T[:3, :3] = rot_m
    T[:3, 3] = trans_vec.flatten()

    return T

def deconstruct_homogenous(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    take a homogenous transformation and split it into rotation matrix and translation vector
    :param T: the homogenous transformation matrix
    :return: the rotation matrix and translation vector
    """
    rot_m = T[:3, :3]
    trans_vec = T[:3, 3]
    return rot_m, trans_vec

def get_samples(data_dir: Path) -> list[Path]:
    """
    Find all directories in the data directory
    :param data_dir: a path object pointing to the data directory
    :return: a list of all sample directories
    """
    return [p for p in data_dir.iterdir() if p.is_dir()]


def get_robot(sample_dir: Path, filename: str="label") -> np.ndarray:
    """
    :param sample_dir: the sample directory
    :param filename: the name of the json file
    :return: The Transformation from robot base to the manipulator (T_b_tcb)
    """
    with open(sample_dir / f"{filename}.json", "r") as json_f:
        data = json.load(json_f)

    T = np.array(data["Transform"])

    if T.shape != (4, 4):
        Exception("Error with Shape")

    return T


def get_transform(
        sample_path: Path,
        camera_matrix: np.ndarray,
        distortion_coefficients: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Takes the given image file, detects the aruco marker in it
    and uses an PNP algorithm to calculate the OpenCV rotation and translation vector
    :param sample_path: the path to the image
    :param camera_matrix: the OpenCV intrinsic calibration camera matrix
    :param distortion_coefficients: the OpenCV intrinsic calibration distortion coefficients
    :return: opencv rotation and translation vector
    """

    marker_size = 100

    object_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)

    image = cv2.imread(str(sample_path))

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)  # SELECT MARKER HERE
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        for i, id in enumerate(ids):
            marker_corners = corners[i][0]

            retval, rvec, tvec = cv2.solvePnP(
                object_points,
                marker_corners,
                camera_matrix,
                distortion_coefficients
            )

    return rvec, tvec


def rotation_from_opencv(rvec: np.ndarray) -> Rotation:
    """
    turns the OpenCV rotation vector into Rotation object
    :param rvec: the opencv rotation vector
    :return: rotation object
    """
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    rot =  Rotation.from_matrix(rotation_matrix)
    return rot


def main(data_dir: Path) -> None:
    camera_matrix, distortion_coefficients = load_calibration(data_dir)

    T_tcb_m = gen_homogenous(
        rot_m = np.zeros((3, 3), dtype=np.float32),
        trans_vec = np.array([[0],
                              [0],
                              [0]], dtype=np.float32)
    )

    sample_dirs = get_samples(data_dir)

    center_rots = []
    center_trans = []

    for sample_dir in sample_dirs:
        T_b_tcb = get_robot(sample_dir)  # load data from the json file

        rots: list[Rotation] = []
        trans: list[np.ndarray] = []

        for source in ["left", "right"]:
            rvec, tvec = get_transform(sample_dir / f"{source}.jpg", camera_matrix, distortion_coefficients)

            rot_debug = rotation_from_opencv(rvec)
            print(rot_debug.as_euler("xyz", degrees=True))
            rots.append(rot_debug)
            trans.append(tvec)

        # calculate the mean of stereo rotations and translation to get the center of both
        rot = Rotation.concatenate(rots).mean()
        tran = np.average(trans, axis=0)

        T_c_m = gen_homogenous(
            rot_m=rot.as_matrix(),
            trans_vec=tran
        )

        # base to marker calculation
        T_b_m = T_b_tcb @ T_c_m

        # base to stereo camera center calculation
        T_b_c = T_b_m @ np.linalg.inv(T_c_m)

        rot_b_c, t_b_c = deconstruct_homogenous(T_b_c)
        center_rots.append(Rotation.from_matrix(rot_b_c))
        center_trans.append(t_b_c)


    center_rot = Rotation.concatenate(center_rots).mean().as_matrix()
    center_tran = np.average(center_trans, axis=0)

    T_c_m = gen_homogenous(center_rot, center_tran)

    print(T_c_m)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    path = args.data_dir

    # path = Path("captured_images")

    main(path)
