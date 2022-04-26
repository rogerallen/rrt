#
# ./anim2.py
# make
# make anim2.mp4
#
BASE=anim2

targets=$(shell for file in `find . -name '*.txt' -type f -printf "%f\n" | sed 's/\..*/\.png/'`; do echo "$$file "; done;)

all: $(targets)

%.png: %.txt
	../../rrt -s 100 -w 1280 -h 720 -i $< -o $@

$(BASE).mp4:
	../../make_mp4 $@

clean:
	rm -f *.txt *.png
