# https://docs.opencv.org/4.9.0/dc/dbb/tutorial_py_calibration.html
import cv2
import numpy as np


def get_obj_img_pts(imgs, row, col):
    CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((row * col, 3), np.float32)
    objp[:, :2] = np.mgrid[0:row, 0:col].T.reshape(-1, 2)

    obj_pts = []
    img_pts = []
    for img in imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (row, col), None)
        if ret is True:
            obj_pts.append(objp)
            cns = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            img_pts.append(cns)

            # show
            cv2.drawChessboardCorners(img, (row, col), cns, ret)
            cv2.imshow("img", img)
            cv2.waitKey()
            cv2.destroyAllWindows()
    return obj_pts, img_pts


imgs = tuple(map(lambda x: cv2.imread(x), ["left.png", "right.png"]))
size = imgs[0].shape[:2][::-1]  # (w, h)
obj_pts, img_pts = get_obj_img_pts(imgs, 4, 6)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)

# print
print("camera matrix:", mtx, "lens distortion:", dist, sep="\n")
for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
    r, _ = cv2.Rodrigues(rvec)
    print(f"rotation matrix-{i}:", r, f"translation vector-{i}:", tvec, sep="\n")

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, size, 1, size)
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, size, 5)

dst = cv2.remap(imgs[0].copy(), mapx, mapy, cv2.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y : y + h, x : x + w]

cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()
