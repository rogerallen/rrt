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

NVCC_FLAGS     = -ccbin $(HOST_COMPILER) -m64 $(NVCC_DBG) $(NVCC_GENCODE)

SRCS = main.cpp rrt.cu
INCS = vec3.h ray.h hittable.h hittable_list.h sphere.h triangle.h camera.h material.h scene.h stb_image_write.h

# default
all: rrt rrtd rrtc

rrt: $(SRCS) $(INCS)
	$(NVCC) $(NVCC_FLAGS) -DFP_T=float -DUSE_CUDA main.cpp rrt.cu -o $@ 

rrtd: $(SRCS) $(INCS)
	$(NVCC) $(NVCC_FLAGS) -DFP_T=double -DUSE_CUDA main.cpp rrt.cu -o $@ 

rrtc: main.cpp rrt.cpp $(INCS)
	$(NVCC) $(NVCC_DBG) -DFP_T=float main.cpp rrt.cpp -o $@

# ??? 
%.png: %.txt
	./rrt -i $< -o $@

%d.png: %.txt
	./rrtd -i $< -o $@

%c.png: %.txt
	./rrtc -i $< -o $@

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
	rm -f rrt rrtc rttd *.ppm *.png *.nsys-rep *.ncu-rep *.log *.sqlite
