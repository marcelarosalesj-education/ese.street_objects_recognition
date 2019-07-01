# Street Mapping Device

Here's the code for the proof of concept of a street mapping device

# Download tarball with GLD testing videos
```
./download_videos.sh
```

# Run all similarity algorithms for all input videos
```
./run_all.sh
```

## Street mapping device tool usage
```
$ python main.py
usage: main.py [-h] -i INPUT [-o OUTPUT] [-sa {sift,kaze,surf,ssim}]
main.py: error: the following arguments are required: -i/--input
$ python main.py -i ../images/example.mp4

```
