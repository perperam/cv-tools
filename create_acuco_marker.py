import cv2 as cv
import cv2.aruco as aruco

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
marker_id = 33
marker_size = 200

marker_image = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

image_file = 'aruco_marker33.png'
cv.imwrite(image_file, marker_image)

print(f"ArUco marker with ID {marker_id} saved to {image_file}")
