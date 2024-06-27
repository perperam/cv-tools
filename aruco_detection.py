import cv2 as cv
import cv2.aruco as aruco
import numpy as np

marker_size = 100

object_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                          [marker_size / 2, marker_size / 2, 0],
                          [marker_size / 2, -marker_size / 2, 0],
                          [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)


fs = cv.FileStorage('camera_parameters.yaml', cv.FILE_STORAGE_READ)
camera_matrix = fs.getNode('camera_matrix').mat()
distortion_coefficients = fs.getNode('distortion_coefficients').mat()
fs.release()

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

parameters = aruco.DetectorParameters()

detector = aruco.ArucoDetector(aruco_dict, parameters)

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)

        for i, id in enumerate(ids):
            marker_corners = corners[i][0]
            marker_id = id[0]

            print(f"Marker ID: {marker_id}")
            retval, rvec, tvec = cv.solvePnP(
                object_points,
                marker_corners,
                camera_matrix,
                distortion_coefficients
            )
            if retval:
                cv.drawFrameAxes(frame, camera_matrix, distortion_coefficients, rvec, tvec, marker_size*2.5, 2)
                distance = np.linalg.norm(tvec)
                print(f"Euclidian distance: {distance}")

    else:
        print("No markers detected.")

    cv.imshow('Detected ArUco markers', frame)

    # exit loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
