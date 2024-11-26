# Calibrating a Stereo Camera to a robot's base
1. Install OpenCV with python `pip install opencv-python` and SkiPy `pip install skipy`
2. Run `inca.py` for an the intrinsic camera calibration. Use a chessboard with its given parameters.
3. Create a directory structure as the following
```txt
<data_dir>
├── camera_parameters.yaml
├── sample01
│   ├── label.json
│   ├── left.png
│   └── right.png
...
└── sample0n
    ├── label.json
    ├── left.png
    └── right.png
```
4. The images should be jpgs containing the `aruco_marker33.png` as real world images.
5. Run `exca.py` on the `<data_dir>` directory: `python exca.py --data_dir <data_dir>`


## Example files
The `label.json` which is T_b_tcb like this:
```json
{
  "Transform": [[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]
}
```

The `camera_parameters.yaml` with OpenCV notation:
```yaml
%YAML:1.0
---
camera_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 597.07793944019829, 0., 349.8752164009328, 0.,
       596.72210333708801, 245.57985470728192, 0., 0., 1. ]
distortion_coefficients: !!opencv-matrix
   rows: 1
   cols: 5
   dt: d
   data: [ -0.015110855828810112, -0.078360530163780959,
       0.003761810688144112, 0.0032740392085114003,
       -0.029627244197296215 ]
```

