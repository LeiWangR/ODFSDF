Before you use the shell script for extract the soundwave, please first make a new folder for storing the extracted soundwave, e.g., `mkdir sampledataset_audio`

Use ffmpeg with python subprocess for extracting wav from video

The basic command to extract audio from a given video:

`ffmpeg -i test.mp4 -ab 160k -ac 2 -ar 44100 -vn audio.wav`

- 160k is the bit rate 160 kbps;
- 2 is the audio channel
- 44100 is the sample rate (44100 Hz)
- vn means no video

Wrap the command into Python code is:
```python
import subprocess
command = "ffmpeg -i C:/test.mp4 -ab 160k -ac 2 -ar 44100 -vn audio.wav"
subprocess.call(command, shell=True)
```
make sure that ffmpeg is a known task, so in your system environment variables, under path, the path to ffmpeg.exe should be listed, or you can just use the full path to the exe in your python code.
