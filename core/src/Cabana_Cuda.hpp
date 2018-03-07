#ifndef CABANA_CUDA_HPP
#define CABANA_CUDA_HPP

#if defined( __NVCC__ )

#include <type_traits>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>

namespace Cabana
{
//---------------------------------------------------------------------------//
// Cuda tag.
struct Cuda {};

//---------------------------------------------------------------------------//
/*!
 * \brief Memory policy for Cuda computations.
 */
template<>
struct MemoryPolicy<Cuda>
{
    //! Allocate array of a number of objects of type T. This will only work
    //! if T is of trivial type (trivially copyable and contiguous).
    template<class T>
    static
    typename std::enable_if<std::is_trivial<T>::value,void>::type
    allocate( T*& ptr, const std::size_t num_t )
    {
        cudaMalloc( (void**) &ptr, num_t * sizeof(T) );
    }

    //! Dellocate an array.
    template<class T>
    static void deallocate( T* ptr )
    {
        cudaFree( ptr );
    }

    //! Copy from one address in the memory space to another in the same
    //! memory space.
    template<class T>
    static void copy( T* destination, const T* source, const std::size_t count )
    {
        cudaMemcpy(
            destination, source, count*sizeof(T), cudaMemcpyDeviceToDevice );
    }
};

//---------------------------------------------------------------------------//
// CudaUVM tag.
struct CudaUVM {};

//---------------------------------------------------------------------------//
/*!
 * \brief Memory policy for Cuda computations with unified-virtual-memory.
 */
template<>
struct MemoryPolicy<CudaUVM>
{
    //! Allocate array of a number of objects of type T. This will only work
    //! if T is of trivial type (trivially copyable and contiguous).
    template<class T>
    static
    typename std::enable_if<std::is_trivial<T>::value,void>::type
    allocate( T*& ptr, const std::size_t num_t )
    {
        cudaMallocManaged( (void**) &ptr, num_t * sizeof(T) );
    }

    //! Dellocate an array.
    template<class T>
    static void deallocate( T* ptr )
    {
        cudaFree( ptr );
    }

    //! Copy from one address in the memory space to another in the same
    //! memory space.
    template<class T>
    static void copy( T* destination, const T* source, const std::size_t count )
    {
        cudaMemcpy(
            destination, source, count*sizeof(T), cudaMemcpyDeviceToDevice );
    }
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end defined( __NVCC__ )

#endif // end CABANA_CUDA_HPP
