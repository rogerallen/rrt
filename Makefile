HOST_COMPILER ?= g++
NVCC          ?= nvcc
NSYS          ?= nsys
NCU           ?= ncu

# Suggest setting this value externally
# gp100  pascal volta turing ga100 ampere
# sm_60  sm_61  sm_70 sm_75  sm_80 sm_86
NVCC_GENCODE ?= -arch sm_86

# select one of these for Debug vs. Release 
#NVCC_DBG       = -g -G --expt-relaxed-constexpr
NVCC_DBG       = -lineinfo -O3 --expt-relaxed-constexpr

#FP_CONTROL = 
#FP_CONTROL = -DUSE_FLOAT_NOT_DOUBLE

NVCC_FLAGS     = -ccbin $(HOST_COMPILER) -m64 $(NVCC_DBG) $(NVCC_GENCODE)

SRCS = main.cu
INCS = vec3.h ray.h hittable.h hittable_list.h sphere.h triangle.h camera.h material.h scene.h stb_image_write.h

# default
all: scenes/test1.png 
#  scenes/test2.png scenes/checkerboard.png scenes/final.png scenes/test1_d.png scenes/test2_d.png scenes/checkerboard_d.png scenes/final_d.png

rrt: $(SRCS) $(INCS)
	$(NVCC) $(NVCC_FLAGS) -o $@ -DFP_T=float main.cu

rrtd: $(SRCS) $(INCS)
	$(NVCC) $(NVCC_FLAGS) -o $@ -DFP_T=double main.cu

rrtc: rrt.cpp $(INCS)
	$(NVCC) $(NVCC_DBG) rrt.cpp -o $@

# ??? 
#%.png: %.txt rtt
#	rm -f $@
#	./rrt -i %.txt -o $@

scenes/test1.png: scenes/test1.txt rrt
	rm -f $@
	./rrt -i scenes/test1.txt -o $@

scenes/test2.png: scenes/test2.txt rrt
	rm -f $@
	./rrt -i scenes/test2.txt -o $@

scenes/checkerboard.png: scenes/checkerboard.txt rrt
	rm -f $@
	./rrt -i scenes/checkerboard.txt -o $@

scenes/final.png: scenes/final.txt rrt
	rm -f $@
	./rrt -i scenes/final.txt -o $@

scenes/test1_d.png: scenes/test1.txt rrtd
	rm -f $@
	./rrtd -i scenes/test1.txt -o $@

scenes/test2_d.png: scenes/test2.txt rrtd
	rm -f $@
	./rrtd -i scenes/test2.txt -o $@

scenes/checkerboard_d.png: scenes/checkerboard.txt rrtd
	rm -f $@
	./rrtd -i scenes/checkerboard.txt -o $@

scenes/final_d.png: scenes/final.txt rrtd
	rm -f $@
	./rrtd -i scenes/final.txt -o $@

# default run args for profiling
RUNARGS ?= -i scenes/test1.txt -w 640 -h 480 -o scenes/test1.png

# to figure out the right device, if not 0 run this:
#   nsys profile --gpu-metrics-device=help
profile_sys: rrt
	$(NSYS) profile --gpu-metrics-device=0 --stats=true --force-overwrite=true -o profile_sys ./rrt $(RUNARGS) > profile_sys.log

# -f for force overwrite, -c 1 for first launch, -set full to control 
profile_kernel: rrt
	$(NCU) -f -k render -c 1 --set full --import-source on -o profile_kernel ./rrt $(RUNARGS) > profile_kernel.log

clean:
	rm -f rrt rttd *.ppm *.png *.nsys-rep *.ncu-rep *.log *.sqlite
