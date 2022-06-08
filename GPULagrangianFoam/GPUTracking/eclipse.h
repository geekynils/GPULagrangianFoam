//!
//! Used to disable proprietary keywords for the eclipse C++ parser.
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __host__
#define __shared__
#endif
