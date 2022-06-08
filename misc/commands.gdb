set args -data debugData
#set args -data debugData
set cuda memcheck on
set breakpoint pending on
b Kernels.cu:calcLambdacnumKernel
#b TrackingKernel.cu:findFacesKernel
