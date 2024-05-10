import cv2
from cv2 import aruco
import numpy as np


def detect_markers(cap, frame_size, aruco_dict, aruco_params, board, mark_count):
    FRAME_COUNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret_corners = []
    ret_ids = []
    ret_frames = []
    for i in range(FRAME_COUNT):
        _, frame = cap.read()
        frame = cv2.resize(frame, frame_size)
        corners, ids, rejected = aruco.detectMarkers(
            frame, aruco_dict, parameters=aruco_params
        )
        corners, ids, _, _ = aruco.refineDetectedMarkers(
            frame, board, corners, ids, rejected
        )
        # sampling
        if i % 100 == 50 and ids is not None and len(ids) == mark_count:
            ret_corners.append(corners)
            ret_ids.append(ids.ravel())
            ret_frames.append(frame)
        if len(corners) > 0:
            aruco.drawDetectedMarkers(frame, corners, ids)
    cap.release()
    return ret_corners, ret_ids, ret_frames


def calibrate_camera_aruco(corners, ids, frame_size, board):
    cali_corners = np.concatenate(
        [np.array(x).reshape(-1, 4, 2) for x in corners], axis=0
    )
    counter = np.array([len(x) for x in ids])
    cali_ids = np.array(ids).ravel()
    mtx_init = np.array(
        [
            [1000.0, 0.0, frame_size[0] / 2.0],
            [0.0, 1000.0, frame_size[1] / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    dist_init = np.zeros((5, 1))
    _, mtx, dist, _, _ = aruco.calibrateCameraAruco(
        cali_corners,
        cali_ids,
        counter,
        board,
        frame_size,
        mtx_init,
        dist_init,
    )
    return mtx, dist


def calibrate_camera_charuco(
    corners, ids, frames, frame_size, board, aruco_mtx, aruco_dist
):
    cali_corners = []
    cali_ids = []
    for xs, ys, frame in zip(corners, ids, frames):
        _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            xs, ys, frame, board, aruco_mtx, aruco_dist
        )
        cali_corners.append(charuco_corners)
        cali_ids.append(charuco_ids)

    _, mtx, dist, _, _ = aruco.calibrateCameraCharuco(
        cali_corners,
        cali_ids,
        board,
        frame_size,
        aruco_mtx,
        aruco_dist,
    )
    return mtx, dist


def calibrate():
    cap = cv2.VideoCapture("ChArUco_board.mp4")
    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2
    FRAME_SIZE = (FRAME_WIDTH, FRAME_HEIGHT)

    aruco_params = aruco.DetectorParameters_create()
    aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

    GRID_X = 5
    GRID_Y = 7
    SQUARE_SIZE = 4
    MARK_COUNT = 17
    BOARD = aruco.CharucoBoard_create(
        GRID_X, GRID_Y, SQUARE_SIZE, SQUARE_SIZE / 2, aruco_dict
    )

    corners, ids, frames = detect_markers(
        cap, FRAME_SIZE, aruco_dict, aruco_params, BOARD, MARK_COUNT
    )
    mtx, dist = calibrate_camera_aruco(corners, ids, FRAME_SIZE, BOARD)
    mtx, dist = calibrate_camera_charuco(
        corners, ids, frames, FRAME_SIZE, BOARD, mtx, dist
    )
    return mtx, dist


mtx, dist = calibrate()
print("camera matrix:", mtx, "lens distortion:", dist, sep="\n")

# aruco video
cap = cv2.VideoCapture("ArUco_marker.mp4")
FRAME_COUNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2
FRAME_SIZE = (FRAME_WIDTH, FRAME_HEIGHT)
MARK_SIZE = 6

# source videos (size 1:1)
src_caps = {i: cv2.VideoCapture(f"video_source/video_{i}.mp4") for i in range(1, 7)}
SRC_SIZES = {
    i: int(x.get(cv2.CAP_PROP_FRAME_WIDTH)) for i, x in src_caps.copy().items()
}


def read_src_videos():
    fs = {}
    for i, x in src_caps.items():
        ret, fs[i] = x.read()
        if not ret:
            x.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, fs[i] = x.read()
    return fs


# output video
out = cv2.VideoWriter(
    "result.avi",
    cv2.VideoWriter.fourcc(*"XVID"),
    24.0,
    FRAME_SIZE,
)

aruco_params = aruco.DetectorParameters_create()
aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
aruco_dict = aruco.Dictionary_get(aruco.DICT_7X7_50)

MARKER = np.array([[3, 3, 0], [-3, 3, 0], [-3, -3, 0], [3, -3, 0]], dtype=float)

for i in range(FRAME_COUNT):
    _, frame = cap.read()
    frame = cv2.resize(frame, FRAME_SIZE)
    src = read_src_videos()

    corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
    if len(corners) > 0:
        ids = ids.flatten()
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARK_SIZE, mtx, dist)
        for id, rvec, tvec in zip(ids, rvecs, tvecs):
            dst_pts, _ = cv2.projectPoints(MARKER, rvec, tvec, mtx, dist)
            src_size = SRC_SIZES[id]
            src_pts = np.array(
                # [left-bottom, right-bottom, right-top, left-top]
                [[0, src_size], [src_size, src_size], [src_size, 0], [0, 0]]
            )
            H, _ = cv2.findHomography(src_pts, dst_pts)
            warped = cv2.warpPerspective(src[id], H, FRAME_SIZE)

            # overlay
            mask = np.zeros(FRAME_SIZE[::-1], dtype="uint8")
            cv2.fillConvexPoly(mask, dst_pts.astype("int32"), (1, 1, 1), cv2.LINE_AA)
            mask = np.dstack([mask] * 3).astype("float")
            warpedMultiplied = cv2.multiply(warped.astype("float"), mask)
            imageMultiplied = cv2.multiply(frame.astype("float"), 1.0 - mask)
            frame = cv2.add(warpedMultiplied, imageMultiplied).astype("uint8")
    out.write(frame)
cap.release()
