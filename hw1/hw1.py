import cv2
import numpy as np
import pafy
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


def cv2ImgAddText(img, text, pos, text_color=(0, 255, 0), text_size=20):
    if isinstance(img, np.ndarray):  # wheter image is opencv type
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font_text = ImageFont.truetype(
        "C:\\Windows\\Fonts\\msjh.ttc", text_size, encoding="utf-8"
    )
    draw.text(pos, text, text_color, font=font_text)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


videos = []

# test_video origin
name = "test_video.mp4"
origin = cv2.VideoCapture(name)
videos.append({"cap": origin, "name": f"{name} origin", "pos": (0, 0)})

# TODO test_video 1
cap = origin
videos.append({"cap": cap, "name": f"{name} 1", "pos": (0, 1)})

# TODO test_video 2
cap = origin
videos.append({"cap": cap, "name": f"{name} 2", "pos": (1, 0)})

# TODO test_video 3
cap = origin
videos.append({"cap": cap, "name": f"{name} 3", "pos": (1, 1)})

# 聖稜-雪山的脊樑© origin
name = "聖稜-雪山的脊樑©"
URL = "https://youtu.be/PHqhEgkGfrs"
origin = cv2.VideoCapture(pafy.new(URL, False, False).getbest(preftype="mp4").url)
videos.append({"cap": origin, "name": f"{name} origin", "pos": (0, 2)})

# TODO 聖稜-雪山的脊樑© 1
cap = origin
videos.append({"cap": cap, "name": f"{name} 1", "pos": (1, 2)})

# output
name = "result_video"
cv2.namedWindow(name)
HEIGHT = 360
WIDTH = 640
RESULT_HEIGHT = HEIGHT * 2
RESULT_WIDTH = WIDTH * 3
big_frame = np.zeros((RESULT_HEIGHT, RESULT_WIDTH, 3), dtype=np.uint8)
out = cv2.VideoWriter(
    f"{name}.avi", cv2.VideoWriter.fourcc(*"XVID"), 1.0, (RESULT_WIDTH, RESULT_HEIGHT)
)
stop = False
while True:
    for i, viedo in enumerate(videos):
        res, frame = viedo["cap"].read()
        if res:
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            frame = cv2ImgAddText(
                frame, viedo["name"], (8, frame.shape[0] - 32), (255, 255, 255)
            )
            row, col = viedo["pos"]
            big_frame[
                row * frame.shape[0] : (row + 1) * frame.shape[0],
                col * frame.shape[1] : (col + 1) * frame.shape[1],
                :,
            ] = frame[:, :, :]
        else:
            stop = True
            break
    if stop:
        break
    out.write(big_frame)
    cv2.imshow(name, big_frame)
    key = cv2.waitKey(1000 // 30)
    if key == 27:
        break
cv2.destroyAllWindows()
for viedo in videos:
    viedo["cap"].release()
