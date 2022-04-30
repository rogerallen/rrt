# Roger's Ray Tracer (rrt)

An updated version of an accelerated CUDA ray-tracer I started at https://github.com/rogerallen/raytracinginoneweekendincuda based on the first book in the https://raytracing.github.io/ series. 

I started this in 2018 and set it aside until 2022 when I taught a class at work and modernized the code to match changes at https://raytracing.github.io/

This is still just a toy renderer and just something I'm working on for personal enjoyment.

## Building

You will need to set the options for the CUDA architecture to compile for in the `NVCC_GENCODE` env variable.  See https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

```
export NVCC_GENCODE="-arch sm_61 -gencode=arch=compute_61,code=sm_61"
make
make scenes/test1.png scenes/test1d.png scenes/test1c.png
```

## Usage

```
Usage: rrt [options]
  -i file.txt         : input scene file
  -o file.png         : output raytraced PNG image (default is PPM to stdout)
  -w <width>          : output image width. (default = 1200)
  -h <height>         : output image height. (800)
  -s <samples>        : number of samples per pixel. (10)
  -tx <num_threads_x> : number of threads per block in x. (8)
  -ty <num_threads_y> : number of threads per block in y. (8)
  -q                  : query devices & cuda info
  -d <device number>  : use this device (default = 0)
```

## Updates

- added commandline arguments
- added PNG output
- use float or double math in renderer
- add a scene description file
- added objects based on triangles

## To Do

- [ ] add motion blur
- [ ] add BVH
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