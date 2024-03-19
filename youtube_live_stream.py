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


URLS = [
    ("https://youtu.be/EOruPDYV1ZI", "基隆 和平島公園"),
    ("https://youtu.be/XSD5ptYisw8", "新北 九份"),
    ("https://youtu.be/Ndo_8RuefH4", "臺北 大稻埕碼頭"),
    ("https://youtu.be/z_fY1pj1VBw", "象山看臺北"),
    ("https://youtu.be/fP4ecxfsJos", "臺北 大佳河濱公園"),
    ("https://youtu.be/jAdVjwU-Lu4", "新北 市碧潭吊橋"),
    ("https://youtu.be/GUCaVR88ZFU", "桃園 石門水庫"),
    ("https://youtu.be/T4b3REvH_x0", "新北 三峽老街"),
    ("https://youtu.be/YkIUZjVlhv4", "桃園 拉拉山遊客中心"),
    ("https://youtu.be/fjhg3gAnMFg", "臺中 高美濕地"),
    ("https://youtu.be/UCG1aXVO8H8", "臺東 多良車站"),
    ("https://youtu.be/Dl1N4LXuDgU", "屏東 墾丁南灣"),
    ("https://youtu.be/dQ7Sd6PGLdA", "臺東 三仙台"),
    ("https://youtu.be/dY2cRNr5Buw", "嘉義 阿里山"),
    ("https://youtu.be/yeoV-wBdoxQ", "臺南 黃金海岸"),
    ("https://youtu.be/xwAWSh35uuw", "新北 淡水漁人碼頭"),
]

caps = []
fps = 100

for url, title in URLS:
    cap = cv2.VideoCapture(pafy.new(url).getbest(preftype="mp4").url)
    if cap is not None:
        fps = min([fps, int(cap.get(cv2.CAP_PROP_FPS))])
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(f"fps: {fps} height: {height} width: {width}\t{title}")
        caps.append((cap, title))

if caps:
    WINDOW_NAME = "youtube live"
    cv2.namedWindow(WINDOW_NAME)
    big_frame = np.zeros(
        ((len(caps) + 3) // 4 * 180, (4 if len(caps) >= 4 else len(caps)) * 320, 3),
        dtype=np.uint8,
    )
    while True:
        for i, cap in enumerate(caps):
            res, frame = cap[0].read()
            if res:
                frame = cv2.resize(frame, (320, 180))
                frame = cv2ImgAddText(
                    frame, cap[1], (8, frame.shape[0] - 24), (255, 255, 255)
                )
                row = i // 4
                col = i % 4
                big_frame[
                    row * frame.shape[0] : (row + 1) * frame.shape[0],
                    col * frame.shape[1] : (col + 1) * frame.shape[1],
                    :,
                ] = frame[:, :, :]
        cv2.imshow(WINDOW_NAME, big_frame)
        key = cv2.waitKey(1000 // fps)
        if key == 27:
            break
    cv2.destroyAllWindows()
    for cap in caps:
        cap[0].release()
