# usage:
#   source setup_rainbow.sh
#

# PATH & LD_LIBRARY_PATH are all setup externally, but when running sudo 
# you need full paths.
export NSYS=/opt/nvidia/nsight-systems/2022.1.1/bin/nsys
export NCU=/usr/local/NVIDIA-Nsight-Compute/ncu

# bah, even with that for ncu you have to do
# sudo make NCU=/usr/local/NVIDIA-Nsight-Compute/ncu profile_kernel

# no chance of multiple ncu users

# which GPU are you compiling for?
# gp100  pascal volta turing ga100 ampere
# sm_60  sm_61  sm_70 sm_75  sm_80 sm_86
export NVCC_GENCODE="-arch sm_75"