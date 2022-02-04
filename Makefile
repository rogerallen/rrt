HOST_COMPILER  = g++
NVCC           = nvcc -ccbin $(HOST_COMPILER)

# select one of these for Debug vs. Release
#NVCC_DBG       = -g -G
NVCC_DBG       =

NVCCFLAGS      = $(NVCC_DBG) -m64 -lineinfo

SRCS = main.cu
INCS = vec3.h ray.h hitable.h hitable_list.h sphere.h camera.h material.h

# default run args
RUNARGS ?= -w 1200 -h 800 -s 100 -tx 8 -ty 8

# gp100  pascal volta turing ga100 ampere
# sm_60  sm_61  sm_70 sm_75  sm_80 sm_86
GENCODE_FLAGS = -arch sm_80

# default
all: out.ppm

rrt: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ main.cu

out.ppm: rrt
	rm -f $@
	./rrt $(RUNARGS) > $@

# to figure out the right device, if not 0 run this:
#   nsys profile --gpu-metrics-device=help
profile_sys: rrt
	nsys profile --gpu-metrics-device=0 --stats=true --force-overwrite=true -o profile_sys ./rrt $(RUNARGS) > profile_sys.log

# -f for force overwrite, -c 1 for first launch, -set full to control 
profile_kernel: rrt
	ncu -f -k render -c 1 --set full --import-source on -o profile_kernel ./rrt $(RUNARGS) > profile_kernel.log

clean:
	rm -f rrt *.ppm *.nsys-rep *.ncu-rep *.log *.sqlite
