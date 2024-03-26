import os
import shutil

from moviepy import editor as mpy
from moviepy.video.fx.speedx import speedx
from moviepy.video.tools.subtitles import SubtitlesClip
import numpy as np
import pyttsx3


class TextToSpeech:
    def __init__(self):
        # Initialize the engine
        self.engine = pyttsx3.init()
        # getting details of current voice
        voices = self.engine.getProperty("voices")
        self.engine.setProperty("voice", voices[0].id)

    def text_to_speech(self, message):
        self.engine.say(message)
        self.engine.runAndWait()

    def text_to_mp3(self, message, mp3file):
        self.engine.save_to_file(message, mp3file)
        self.engine.runAndWait()


SOURCE_NAME = "result_video.avi"
BGM_NAME = "background_music.wav"
TARGET_NAME = "demo_video.mp4"

FONT = "msjhbd.ttc"
SUBTITLES = ""
with open("subtitles.txt", "r", encoding="utf-8") as f:
    SUBTITLES = f.read()

lines = [msg for msg in SUBTITLES.split("\n") if len(msg) > 0]
subs_audio_clips = []

tts = TextToSpeech()
for i, msg in enumerate(lines):
    subtitle_name = "subtitle{:04d}.mp3".format(i)
    tts.text_to_mp3(msg, subtitle_name)
    subs_audio_clips.append(mpy.AudioFileClip(subtitle_name))

# 計算每一句旁白開始與結束時間，假設開始時間為0。
duration = np.array([0] + [s.duration for s in subs_audio_clips])
cum_d = np.cumsum(duration)
total_d = int(cum_d[-1]) + 4
print(f"total time: {total_d} sec")

shutil.copyfile(f"..\\fonts\\{FONT}", FONT)
generator = lambda txt: mpy.TextClip(txt, font=FONT, fontsize=32, color="white")
subs = [((cum_d[i], cum_d[i + 1]), s) for i, s in enumerate(lines)]
subs = SubtitlesClip(subs, generator)

with mpy.VideoFileClip(SOURCE_NAME) as clip:
    clip = clip.fx(speedx, clip.duration / total_d)

    # subtitles
    final_clip = mpy.CompositeVideoClip([clip, subs.set_pos(("center", "bottom"))])

    # audio
    bgm_clip = mpy.AudioFileClip(BGM_NAME)
    bgm = bgm_clip.subclip(0, total_d).volumex(0.2)
    # (bgm_clip.duration - total_d).volumex(0.2)
    speech = mpy.concatenate_audioclips(subs_audio_clips)
    audio = [clip.audio, bgm, speech]
    audio_clip = mpy.CompositeAudioClip(list(filter(lambda x: x is not None, audio)))

    # save
    final = final_clip.set_audio(audio_clip)
    final.write_videofile(TARGET_NAME)

    bgm_clip.close()
    audio_clip.close()
    final_clip.close()
for x in subs_audio_clips:
    os.remove(x.filename)
    x.close()
subs.close()
os.remove(FONT)
