HOST_COMPILER ?= g++
NVCC          ?= nvcc
NSYS          ?= nsys
NCU           ?= ncu

# Suggest setting this value externally
# gp100  pascal volta turing ga100 ampere
# sm_60  sm_61  sm_70 sm_75  sm_80 sm_86
NVCC_GENCODE ?= -arch sm_86

# select one of these for Debug vs. Release 
#NVCC_DBG       = -g -G
NVCC_DBG       = -lineinfo

NVCC_FLAGS     = -ccbin $(HOST_COMPILER) -m64 $(NVCC_DBG) $(NVCC_GENCODE) 

SRCS = main.cu
INCS = vec3.h ray.h hitable.h hitable_list.h sphere.h camera.h material.h scene.h stb_image_write.h

# default
all: scenes/test1.png scenes/final.png

rrt: $(SRCS) $(INCS)
	$(NVCC) $(NVCC_FLAGS) -o $@ main.cu

scenes/test1.png: rrt
	rm -f $@
	./rrt -i scenes/test1.txt -o $@

scenes/final.png: rrt
	rm -f $@
	./rrt -i scenes/final.txt -s 100 -o $@

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
	rm -f rrt *.ppm *.png *.nsys-rep *.ncu-rep *.log *.sqlite
