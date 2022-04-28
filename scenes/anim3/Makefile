#
# ./anim3.py
# make
# make anim3.mp4
#
BASE=anim3

targets=$(shell for file in `find . -name '*.txt' -type f -printf "%f\n" | sed 's/\..*/\.png/'`; do echo "$$file "; done;)

all: $(targets)

%.png: %.txt
	../../rrt -s 10 -w 1280 -h 720 -i $< -o $@

$(BASE).list: anim3list.py
	./anim3list.py > $@

$(BASE).mp4: $(BASE).list
	ffmpeg \
	  -y -f concat -i $(BASE).list \
	  -pix_fmt yuv420p -an \
	  -c:v libx264 -preset veryslow -b:v 5000k -pass 1 \
	  -r 30 \
	  -f mp4 /dev/null && \
	ffmpeg \
	  -f concat -i $(BASE).list \
	  -pix_fmt yuv420p -an \
	  -c:v libx264 -preset veryslow -b:v 5000k -pass 2 \
	  -r 30 $@

clean:
	rm -f *.txt *.png
