# Roger's Ray Tracer (rrt)

## To Do

- [x] Copy from raytracinginoneweekendincude nvidia branch
- [x] add prefetch to/from GPU
- [x] real commandline options
- [x] scene loading
- [x] PNG output
- [x] retry using vec3 in scene.h?
- [x] double precision calculations
- [x] select device
- [x] shared-memory sample reduction
- [x] triangle-based objects
- [x] modernize code from class
- [x] split main.cu into main.cpp & rrt.cu
- [x] add PNG to cpp path
- [x] add scene handling to rrtc
- [ ] input list of scenes to render
- [ ] cmake build?
- [ ] add pybind11 to enable running from python

## Status

Computelab: everything works
Rainbow:    sudo /opt/nvidia/nsight-systems/2022.1.1/bin/nsys    segfaults
            sudo /opt/nvidia/nsight-systems/2022.1.1/bin/nsys-ui works(!)
            /opt/nvidia/nsight-compute/2022.1.1/ncu              works
            /opt/nvidia/nsight-compute/2022.1.1/ncu-ui           works
Gyre (WSL): learned async memcopy doesn't work. nsys & ncu TBD