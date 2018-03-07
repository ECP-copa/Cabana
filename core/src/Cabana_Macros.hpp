#ifndef CABANA_MACROS_HPP
#define CABANA_MACROS_HPP

#if defined(__NVCC__)

#define CABANA_INLINE_FUNCTION __host__ __device__ inline
#define CABANA_FUNCTION __host__ __device__

#else

#define CABANA_INLINE_FUNCTION inline
#define CABANA_FUNCTION

#endif

#endif // end CABANA_MACROS_HPP
