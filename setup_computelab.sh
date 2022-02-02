# usage:
#   source computelab_setup.sh
#
export CUDA_VERSION=11.5
export CUDA_HOME=/home/scratch.svc_compute_arch/release/cuda_toolkit/r$CUDA_VERSION/x86_64/latest
export NSIGHT_SYSTEMS_HOME=/home/scratch.svc_compute_arch/release/nsightSystems/x86_64/rel/2021.5.1.118/bin
export NSIGHT_COMPUTE_HOME=/home/scratch.svc_compute_arch/release/nsightCompute/cuda_11.5/x86_64/2021.3.0.9

export PATH=$CUDA_HOME/bin:$NSIGHT_SYSTEMS_HOME:$NSIGHT_COMPUTE_HOME:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/nvvm/lib64:$LD_LIBRARY_PATH

# make ncu happy for multiple users
export TMPDIR=/tmp/$USER
rm -rf $TMPDIR
mkdir -p $TMPDIR