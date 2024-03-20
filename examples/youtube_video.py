import time

import cv2
import pafy


URL = "https://www.youtube.com/watch?v=h8DLofLM7No"
yt_video = pafy.new(URL, basic=False, gdata=False)
best = yt_video.getbest("mp4")
cap = cv2.VideoCapture(best.url)

if cap is not None:
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"fps: {FPS}\nheight: {HEIGHT}\nwidth: {WIDTH}")
    TOTAL_FRAME = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0
    play_flag = 1

    def set_frame_number(x):
        global frame_num, play_flag
        if play_flag == 0:
            frame_num = x
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        return

    def play(x):
        global play_flag
        play_flag = x
        return

    cv2.namedWindow("youtube")
    cv2.createTrackbar("frame no.", "youtube", 0, TOTAL_FRAME - 1, set_frame_number)
    cv2.createTrackbar("play", "youtube", 0, 1, play)
    cv2.setTrackbarPos("play", "youtube", play_flag)

    while True:
        cv2.setTrackbarPos("frame no.", "youtube", frame_num)
        t1 = time.perf_counter_ns()
        if play_flag:
            res, frame = cap.read()
            if res:
                cv2.imshow("youtube", frame)
                frame_num += 1
            else:
                break
        t2 = 1000 // FPS - (time.perf_counter_ns() - t1) // 1000000
        key = cv2.waitKey(max(t2, 1))
        if key == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
else:
    print(f"cannot open {URL}")
