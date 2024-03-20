import os

import cv2


cap = cv2.VideoCapture("sample_video.mp4")
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(f"height: {HEIGHT}\nwidth: {WIDTH}")
TOTAL_FRAME = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_num = 0


def set_frame_number(x):
    global frame_num
    frame_num = x
    return


# show
cv2.namedWindow("video file")
cv2.createTrackbar("frame no.", "video file", 0, TOTAL_FRAME - 1, set_frame_number)

while frame_num < TOTAL_FRAME:
    cv2.setTrackbarPos("frame no.", "video file", frame_num)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    res, frame = cap.read()
    if not res:
        break
    cv2.imshow("video file", frame)
    key = cv2.waitKey(10) & 0xFF
    if key == 27:  # esc
        break
    frame_num += 1
cv2.destroyAllWindows()

# write
PATH = "result"
if not os.path.exists(PATH):
    os.mkdir(PATH)
out = cv2.VideoWriter(
    f"{PATH}/video.avi", cv2.VideoWriter.fourcc(*"XVID"), 10.0, (WIDTH, HEIGHT)
)
print("writting video...")
while True:
    res, frame = cap.read()
    if not res:
        break
    cv2.rectangle(
        frame,
        (frame.shape[1] // 2 - 50, frame.shape[0] // 2 - 50),
        (frame.shape[1] // 2 + 50, frame.shape[0] // 2 + 50),
        (0, 0, 255),
    )
    out.write(frame)
    # cv2.imshow("capture", frame)
print("done.")
cap.release()
