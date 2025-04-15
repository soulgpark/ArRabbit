import cv2
import numpy as np

chessboard_size = (6, 9)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

data = np.load('camera/data/calibration_result.npz')
mtx, dist = data['mtx'], data['dist']

rabbit_pts_3d = {
    "face_center":     [4.5, 1.5, 0],
    "left_ear_top":    [4.3, 1.3, -0.8],
    "left_ear_base1":  [4.25, 1.4, -0.3],
    "left_ear_base2":  [4.35, 1.4, -0.3],
    "right_ear_top":   [4.7, 1.3, -0.8],
    "right_ear_base1": [4.65, 1.4, -0.3],
    "right_ear_base2": [4.75, 1.4, -0.3],
    "left_eye":        [4.4, 1.6, -0.1],
    "right_eye":       [4.6, 1.6, -0.1],
}

cap = cv2.VideoCapture('camera/data/chessboard.avi')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("camera/data/ar_rabbit_result.mp4", fourcc, fps, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret_corners:
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        ret_pnp, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)

        if ret_pnp:
            pts_3d = np.array(list(rabbit_pts_3d.values()), dtype=np.float32)
            imgpts, _ = cv2.projectPoints(pts_3d, rvec, tvec, mtx, dist)
            imgpts = imgpts.reshape(-1, 2).astype(int)

            (
                face_center,
                le_top, le_b1, le_b2,
                re_top, re_b1, re_b2,
                leye, reye
            ) = imgpts

            frame = cv2.circle(frame, tuple(face_center), 30, (255, 200, 200), -1)

            frame = cv2.circle(frame, tuple(leye), 5, (0, 0, 0), -1)
            frame = cv2.circle(frame, tuple(reye), 5, (0, 0, 0), -1)

            cv2.line(frame, tuple(le_top), tuple(le_b1), (255, 150, 200), 8)
            cv2.line(frame, tuple(le_top), tuple(le_b2), (255, 150, 200), 8)
            cv2.line(frame, tuple(le_b1), tuple(le_b2), (255, 150, 200), 8)

            cv2.line(frame, tuple(re_top), tuple(re_b1), (255, 150, 200), 8)
            cv2.line(frame, tuple(re_top), tuple(re_b2), (255, 150, 200), 8)
            cv2.line(frame, tuple(re_b1), tuple(re_b2), (255, 150, 200), 8)

    out.write(frame)
    cv2.imshow("AR Rabbit", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
