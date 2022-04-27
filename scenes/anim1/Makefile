#
# ./anim1.py
# make
# make anim1.mp4
#
targets=$(shell for file in `find . -name '*.txt' -type f -printf "%f\n" | sed 's/\..*/\.png/'`; do echo "$$file "; done;)

all: $(targets)

%.png: %.txt
	../../rrt -s 100 -i $< -o $@

anim1.mp4:
	../../make_mp4 anim1.mp4

clean:
	rm -f *.txt *.png
