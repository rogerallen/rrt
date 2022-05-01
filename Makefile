HOST_COMPILER ?= g++
NVCC          ?= nvcc
NSYS          ?= nsys
NCU           ?= ncu

# See https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# Also, add -DCOMPILING_FOR_WSL if you are compiling on Windows Subsystem for Linux
NVCC_GENCODE ?= -arch sm_86

# select one of these for Debug vs. Release 
#NVCC_DBG       = -g -G --expt-relaxed-constexpr
NVCC_DBG       = -lineinfo -O3 --expt-relaxed-constexpr

NVCC_FLAGS     = -ccbin $(HOST_COMPILER) -m64 $(NVCC_DBG) $(NVCC_GENCODE)

SRCS = main.cpp rrt.cu
SRCSC = main.cpp rrt.cpp
INCS = rrt.h vec3.h ray.h hittable.h hittable_list.h sphere.h triangle.h camera.h material.h scene.h stb_image_write.h

# default
all: rrt rrtd rrtc

# main binaries
rrt: $(SRCS) $(INCS)
	$(NVCC) $(NVCC_FLAGS) -DFP_T=float -DUSE_CUDA $(SRCS) -o $@ 

rrtd: $(SRCS) $(INCS)
	$(NVCC) $(NVCC_FLAGS) -DFP_T=double -DUSE_CUDA $(SRCS) -o $@ 

rrtc: $(SRCSC) $(INCS)
	$(NVCC) $(NVCC_DBG) -DFP_T=float $(SRCSC) -o $@

# output images for float CUDA, double CUDA & C++
%.png: %.txt
	./rrt -i $< -o $@

%d.png: %.txt
	./rrtd -i $< -o $@

%c.png: %.txt
	./rrtc -i $< -o $@

# some hints on running nsight systems & nsight compute
# default run args for profiling
RUNARGS ?= -s 1 -i scenes/test1.txt -w 640 -h 480 -o scenes/test1.png

# to figure out the right device, if not 0 run this:
#   nsys profile --gpu-metrics-device=help
profile_sys: rrt
	$(NSYS) profile --gpu-metrics-device=0 --stats=true --force-overwrite=true -o profile_sys ./rrt $(RUNARGS) > profile_sys.log

# -f for force overwrite, -c 1 for first launch, -set full to control 
profile_kernel: rrt
	$(NCU) -f -k render -c 1 --set full --import-source on -o profile_kernel ./rrt $(RUNARGS) > profile_kernel.log

clean:
	rm -f rrt rrtc rrtd *.ppm *.png *.nsys-rep *.ncu-rep *.log *.sqlite
