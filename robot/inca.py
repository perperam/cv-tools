"""
Author: https://github.com/perperam
# using parts of this tutorial
# https://docs.opencv.org/4.9.0/dc/dbb/tutorial_py_calibration.html
# accessed 2024-06-20
"""

import cv2 as cv
import numpy as np
from pathlib import Path


def main():
    grid_width = int(input("Enter the number of internal corners in width (grid size): "))
    grid_height = int(input("Enter the number of internal corners in height (grid size): "))
    checker_size = float(input("Enter the checker size in mm: "))
    take_new_images = True if input("Do you want to take new images (y/n): ") == 'y' else False

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((grid_height * grid_width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_width, 0:grid_height].T.reshape(-1, 2) * checker_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    images_dir = Path('./images')
    images_dir.mkdir(parents=True, exist_ok=True)

    if take_new_images:
        cap = cv.VideoCapture(0)
        img_counter = 0

        print("Press SPACE to take a new image and ENTER to finish recording")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv.imshow('Camera', frame)

            k = cv.waitKey(1)
            if k % 256 == 32:  # SPACE pressed
                img_name = images_dir / f"cap_{img_counter:02d}.jpg"
                cv.imwrite(str(img_name), frame)
                print(f"{img_name} written!")
                img_counter += 1
            elif k % 256 == 13:  # ENTER pressed
                break

        cap.release()
        cv.destroyAllWindows()

    image_paths = list(images_dir.glob('cap_*.jpg'))

    for img_path in image_paths:
        img = cv.imread(str(img_path))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, (grid_width, grid_height), None)

        # if found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # draw and display the corners
            cv.drawChessboardCorners(img, (grid_width, grid_height), corners2, ret)
            cv.imshow('Checkerboard', img)
            cv.waitKey(500)

    cv.destroyAllWindows()

    # perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


    y_file = cv.FileStorage('camera_parameters.yaml', cv.FILE_STORAGE_WRITE)
    y_file.write('camera_matrix', mtx)
    y_file.write('distortion_coefficients', dist)
    y_file.release()

    print("Camera matrix:")
    print(mtx)
    print("Distortion coefficients:")
    print(dist)

if __name__ == '__main__':
    main()
