# usage:
#   source setup_gyre.sh
#

# setup PATH & LD_LIBRARY_PATH
#export CUDA_VERSION=11.5
export CUDA_HOME=/usr/local/cuda
# ? export NSIGHT_SYSTEMS_HOME=/home/scratch.svc_compute_arch/release/nsightSystems/x86_64/rel/2021.5.1.118/bin
# ? export NSIGHT_COMPUTE_HOME=/home/scratch.svc_compute_arch/release/nsightCompute/cuda_11.5/x86_64/2021.3.0.9

#export PATH=$CUDA_HOME/bin:$NSIGHT_SYSTEMS_HOME:$NSIGHT_COMPUTE_HOME:$PATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/nvvm/lib64:$LD_LIBRARY_PATH

# make ncu happy for multiple users
#export TMPDIR=/tmp/$USER
#rm -rf $TMPDIR
#mkdir -p $TMPDIR

# which GPU are you compiling for?
# gp100  pascal volta turing ga100 ampere
# sm_60  sm_61  sm_70 sm_75  sm_80 sm_86
export NVCC_GENCODE="-DCOMPILING_FOR_WSL -arch sm_86"
