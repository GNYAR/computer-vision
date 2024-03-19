# computer-vision

## Environment

- Windows 10
- Python 3.11.0
- CUDA 12.1

```bash
# matplotlib numpy opencv-python
pip install matplotlib numpy opencv-python
# pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

- opencv-python 4.9.0.80 ([doc](https://docs.opencv.org/4.9.0/))
- torch 2.2.1+cu121 ([doc](https://pytorch.org/docs/2.2/))
- torchaudio 2.2.1+cu121
- torchvision 0.17.1+cu121

```bash
# youtube-dl pafy
pip install youtube-dl
pip install git+https://github.com/Cupcakus/pafy.git

python -c "import os; print('\\'.join(os.sys.path[1].split('\\')[:-1]))"
# The output is your PYTHON_PATH.
# Please follow the steps blew to fix problem.
# Add parameter flatal=Flase in youtube.py line 1794.
code PYTHON_PATH\Lib\site-packages\youtube_dl\extractor\youtube.py
```

- pafy ([source](https://github.com/Cupcakus/pafy))
