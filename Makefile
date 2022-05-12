HOST_COMPILER ?= g++
NVCC          ?= nvcc
NSYS          ?= nsys
NCU           ?= ncu

# See https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# Also, add -DCOMPILING_FOR_WSL if you are compiling on Windows Subsystem for Linux
NVCC_GENCODE ?= -arch sm_86

# select one of these for Debug vs. Release
# add -Werror when you can
#GCC_WARNINGS   = -pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5 -Wswitch-default -Wundef -Wno-unused -Werror 
GCC_WARNINGS   = 
HOST_DBG       = -g $(GCC_WARNINGS)
HOST_REL       = -O3 $(GCC_WARNINGS)
#NVCC_DBG       = $(HOST_DBG) -G --expt-relaxed-constexpr
NVCC_DBG       = -lineinfo $(HOST_REL) --expt-relaxed-constexpr

NVCC_FLAGS     = -ccbin $(HOST_COMPILER) -m64 $(NVCC_DBG) $(NVCC_GENCODE)

SRCS = main.cpp rrt.cu
SRCSC = main.cpp rrt.cpp
INCS = rrt.h scene.h \
	aabb.h bvh.h camera.h hittable.h hittable_list.h material.h moving_sphere.h ray.h sphere.h triangle.h vec3.h \
	rtweekend.h stb_image_write.h

# default
all: rrt rrtd rrtc rrto

# main binaries
# 1) CUDA with single-precision
rrt: $(SRCS) $(INCS)
	$(NVCC) $(NVCC_FLAGS) -DUSE_FLOAT -DUSE_CUDA $(SRCS) -o $@ 

# 2) CUDA with double-precision
rrtd: $(SRCS) $(INCS)
	$(NVCC) $(NVCC_FLAGS) -DUSE_CUDA $(SRCS) -o $@ 

# 3) CPU with single-precision
# to be fair to CPU, adjust this to use double, but using float catches bugs during development, so...
rrtc: $(SRCSC) $(INCS)
	$(HOST_COMPILER) $(HOST_REL) -DUSE_FLOAT $(SRCSC) -o $@

# 3) CPU with OpenMP & double-precision
rrto: $(SRCSC) $(INCS)
	$(HOST_COMPILER) $(HOST_REL) -fopenmp $(SRCSC) -o $@ -lgomp

# output images for float CUDA, double CUDA & C++
%.png: %.txt rrt
	./rrt -i $< -o $@

%d.png: %.txt rrtd
	./rrtd -i $< -o $@

%c.png: %.txt rrtc
	./rrtc -i $< -o $@

%o.png: %.txt rrto
	./rrto -i $< -o $@
	
# some hints on running nsight systems & nsight compute
# default run args for profiling
RUNARGS ?= -s 20 -tx 640 -ty 1 -w 640 -h 480 -i scenes/test2.txt -o scenes/test2.png

# to figure out the right device, if not 0 run this:
#   nsys profile --gpu-metrics-device=help
profile_sys: rrt
	$(NSYS) profile --gpu-metrics-device=0 --stats=true --force-overwrite=true -o profile_sys ./rrt $(RUNARGS) > profile_sys.log

# -f for force overwrite, -c 1 for first launch, -set full to control 
profile_kernel: rrt
	$(NCU) -f -k render -c 1 --set full --import-source on -o profile_kernel ./rrt $(RUNARGS) > profile_kernel.log

clean:
	rm -f rrt rrtc rrtd *.ppm *.png *.nsys-rep *.ncu-rep *.log *.sqlite
