# how to

1. make all the scene description files (requires python3)
```
$ ./anim.py
```

2. make all the png frames from the scene files.  This will take a long time--3.3 hours for me.
```
make
```

3. make the movie from the png files  (requires ffmpeg)
```
make final_anim.mp4
```