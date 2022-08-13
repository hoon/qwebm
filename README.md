# qwebm - work in progress

## Description
A wrapper for ffmpeg that makes two-pass encoding easier. Will only convert to WebM container format.

## Usage
Simple usage with default options:
```shell
python qwebm.py <input_video_file>
```

To target a specific size (default is 6 MB):
```shell
python qwebm.py -s 2560 KB <input_video_file>
```

To resize video to specific dimension:
```shell
python qwebm.py --dim 480p <input_video_file>
```

To encode using VP9 codec (default is VP8):
```shell
python qwebm.py --cv vp9 <input_video_file>
```

To print ffmpeg execution arguments `qwebm` would have used but not actually execute ffmpeg:
```shell
python qwebm.py --print-args --nx <input_video_file>
```
