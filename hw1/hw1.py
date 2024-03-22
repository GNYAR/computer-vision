import cv2
import numpy as np
import pafy
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from torchvision.transforms import v2


def cv2ImgAddText(img, text, pos, text_color=(0, 255, 0), text_size=20):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font_text = ImageFont.truetype("..\\fonts\\msjh.ttc", text_size, encoding="utf-8")
    draw.text(pos, text, text_color, font=font_text)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


videos = []

# get test_video
name = "test_video.mp4"
origin = cv2.VideoCapture(name)
videos.append((origin, name))

# get 聖稜-雪山的脊樑©
name = "聖稜-雪山的脊樑©"
URL = "https://youtu.be/PHqhEgkGfrs"
origin = cv2.VideoCapture(pafy.new(URL, False, False).getbest(preftype="mp4").url)
videos.append((origin, name))

# output
RESULT_NAME = "result_video"
cv2.namedWindow(RESULT_NAME)
HEIGHT = 360
WIDTH = 640
RESULT_HEIGHT = HEIGHT * 2
RESULT_WIDTH = WIDTH * 3
out = cv2.VideoWriter(
    f"{RESULT_NAME}.avi",
    cv2.VideoWriter.fourcc(*"XVID"),
    5.0,
    (RESULT_WIDTH, RESULT_HEIGHT),
)
big_frame = np.zeros((RESULT_HEIGHT, RESULT_WIDTH, 3), dtype=np.uint8)


def add_to_big_frame(frame, name, row, col):
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame = cv2ImgAddText(frame, name, (8, frame.shape[0] - 32), (255, 255, 255))
    global big_frame
    big_frame[
        row * frame.shape[0] : (row + 1) * frame.shape[0],
        col * frame.shape[1] : (col + 1) * frame.shape[1],
        :,
    ] = frame[:, :, :]


while True:
    # test_video
    cap, name = videos[0]
    res, frame = cap.read()
    if res:
        add_to_big_frame(frame, name, 0, 0)

        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        add_to_big_frame(hsv_img, "convert to HSV color space", 0, 1)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        blurrer = v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.5, 9.0))
        blurred_img = cv2.cvtColor(np.asarray(blurrer(img)), cv2.COLOR_RGB2BGR)
        add_to_big_frame(blurred_img, "torchvision v2 gaussian blur", 1, 0)

        gamma = 0.5
        gamma_img = np.array(255 * (frame / 255) ** gamma, dtype="uint8")
        add_to_big_frame(gamma_img, f"gamma({gamma}) transformed", 1, 1)
    else:
        break

    # 聖稜-雪山的脊樑©
    cap, name = videos[1]
    res, frame = cap.read()
    if res:
        add_to_big_frame(frame, name, 0, 2)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        gray_img = cv2.cvtColor(np.asarray(v2.Grayscale()(img)), cv2.COLOR_RGB2BGR)
        add_to_big_frame(gray_img, "torchvision v2 gray scale", 1, 2)
    else:
        break

    out.write(big_frame)
    cv2.imshow(RESULT_NAME, big_frame)
    key = cv2.waitKey(10)
    if key == 27:
        break
cv2.destroyAllWindows()
videos[0][0].release()
videos[1][0].release()
