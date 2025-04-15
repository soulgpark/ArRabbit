# ArRabbit
This project estimates the camera pose from camera calibration results and visualizes an AR rabbit on top of a chessboard image.

## Features
- **camera calibration results**: Use internal and external parameters of `calibration_result.npz`
- **Camera Pose Estimation**: Estimate camera rotation vector and translation vector with `cv2.solvePnP()`
- **Visualize AR rabbit shape**


  -Face: Pink circle

  
  -Eyes: Black dots

  
  -Ears: Triangular shape, emphasize three-dimensionality

## Ar rabbit output
![image](https://github.com/user-attachments/assets/b05b0692-b9c6-450b-8e8b-0280c76429de)

