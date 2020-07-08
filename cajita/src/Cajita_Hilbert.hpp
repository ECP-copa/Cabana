#ifndef HILBERT_HPP
#define HILBERT_HPP

#ifndef DEBUG
    #define DEBUG 0 
#endif 

#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>

namespace Kokkos
{

    struct LayoutHilbert {
        typedef LayoutHilbert array_layout;

        size_t dimension[ARRAY_LAYOUT_MAX_RANK];

        enum { is_extent_constructible = true };

        LayoutHilbert( LayoutHilbert const & ) = default;
        LayoutHilbert( LayoutHilbert && ) = default;
        LayoutHilbert& operator=( LayoutHilbert const & ) = default;
        LayoutHilbert& operator=( LayoutHilbert && ) = default;

        KOKKOS_INLINE_FUNCTION
        explicit constexpr LayoutHilbert( size_t N0 = 0, size_t N1 = 0, size_t N2 = 0,
                                            size_t N3 = 0, size_t N4 = 0, size_t N5 = 0,
                                            size_t N6 = 0, size_t N7 = 0 )
        : dimension{ N0, N1, N2, N3, N4, N5, N6, N7 }
        {}

    };

    namespace Impl
    {
        struct HilbertMap2D {
            int hilbert_step;
            Kokkos::View<int*> map;

            HilbertMap2D() {};

            HilbertMap2D( long unsigned int width, long unsigned int height ) {
                // std::cout << "HilbertMap2D: " << width << " " << height << "\n";
                if ( width > 0 && height > 0 ) {
                    hilbert_step = 0;
                    Kokkos::resize( map, width * height );

                    if ( width >= height ) {
                        gilbert2d( 0, 0, width, 0, 0, height, width, height );
                    }
                    else {
                        gilbert2d( 0, 0, 0, height, width, 0, width, height );
                    }
                }
            };

            void gilbert2d( int x, int y, int ax, int ay, int bx, int by, int nx, int ny ) {
                int w = std::abs( ax + ay );
                int h = std::abs( bx + by );

                int dax = ( ( ax > 0 ) - ( ax < 0 ) );
                int day = ( ( ay > 0 ) - ( ay < 0 ) );
                int dbx = ( ( bx > 0 ) - ( bx < 0 ) );
                int dby = ( ( by > 0 ) - ( by < 0 ) );

                if ( h == 1 ) {
                    for ( int i = 0; i < w; i++ ) {
                        if ( DEBUG ) std::cout << "( " << x << ", " << y << " )\t" << x + nx * y << "\t" << hilbert_step << "\n";
                        map( x + nx * y ) = hilbert_step;
                        hilbert_step ++;
                        x += dax;
                        y += day;
                    }
                    return;
                }

                if ( w == 1 ) {
                    for ( int i = 0; i < h; i++ ) {
                        if ( DEBUG ) std::cout << "( " << x << ", " << y << " )\t" << x + nx * y << "\t" << hilbert_step << "\n";
                        map( x + nx * y ) = hilbert_step;
                        hilbert_step ++;
                        x += dbx;
                        y += dby;
                    }
                    return;
                }

                int ax2 = ax / 2;
                int ay2 = ay / 2;
                int bx2 = bx / 2;
                int by2 = by / 2;

                int w2 = std::abs( ax2 + ay2 );
                int h2 = std::abs( bx2 + by2 );

                if ( 2 * w > 3 * h ) {
                    if ( ( w2 % 2 ) && ( w > 2 ) ) {
                        ax2 += dax;
                        ay2 += day;
                    }

                    gilbert2d( x, y, ax2, ay2, bx, by, nx, ny );
                    gilbert2d( x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by, nx, ny );
                }
                else {
                    if ( ( h2 % 2 ) && ( h > 2 ) ) {
                        bx2 += dbx;
                        by2 += dby;
                    }

                    gilbert2d( x, y, bx2, by2, ax2, ay2, nx, ny );
                    gilbert2d( x + bx2, y + by2, ax, ay, bx - bx2, by - by2, nx, ny );
                    gilbert2d( x + ( ax - dax ) + ( bx2 - dbx ), y + ( ay - day ) + ( by2 - dby ), -bx2, -by2, - ( ax - ax2 ), - ( ay - ay2 ), nx, ny );
                }
            };
        };

        template <class Dimension>
        struct ViewOffset<Dimension, Kokkos::LayoutHilbert, void> {

            using is_mapping_plugin = std::true_type;
            using is_regular        = std::true_type;

            typedef size_t size_type;
            typedef Dimension dimension_type;
            typedef Kokkos::LayoutHilbert array_layout;

            dimension_type m_dim;
            int sub_flag = 0;

            HilbertMap2D hilbert_map{ m_dim.N0, m_dim.N1 };

            // rank 1
            template <typename I0>
            KOKKOS_INLINE_FUNCTION 
            constexpr size_type operator()( I0 const& i0 ) const {
                return i0;
            };

            // rank 2
            template <typename I0, typename I1>
            KOKKOS_INLINE_FUNCTION 
            constexpr size_type operator()( I0 const& i0, I1 const& i1 ) const {
                if ( sub_flag ) return i1 + m_dim.N1 * i0;
                else {
                    // std::cout << i0 + m_dim.N0 * i1 << "\t";

                    int hilbert = hilbert_map.map( i0 + m_dim.N0 * i1 );

                    // std::cout << "Hilbert Index: " << hilbert << "\t";

                    return hilbert;
                }
            };

            // rank 3
            template <typename I0, typename I1, typename I2>
            KOKKOS_INLINE_FUNCTION 
            constexpr size_type operator()( I0 const& i0, I1 const& i1, I2 const& i2 ) const {
                if ( sub_flag ) return i2 + m_dim.N2 * ( i1 + m_dim.N1 * ( i0 ) );
                else {
                    // std::cout << i0 + m_dim.N0 * ( i1 + m_dim.N1 * i2 ) << "\t";
                    
                    int hilbert = hilbert_map.map( i0 + m_dim.N0 * i1 );

                    // std::cout << "Hilbert Index: " << m_dim.N0 * m_dim.N1 * i2 + hilbert << "\t";

                    return m_dim.N0 * m_dim.N1 * i2 + hilbert;
                }
            };

            // rank 4
            template <typename I0, typename I1, typename I2, typename I3>
            KOKKOS_INLINE_FUNCTION 
            constexpr size_type operator()( I0 const& i0, I1 const& i1, I2 const& i2, I3 const& i3 ) const {
                if ( sub_flag ) return i3 + m_dim.N3 * ( i2 + m_dim.N2 * ( i1 + m_dim.N1 * ( i0 ) ) );
                else {
                    // std::cout << i0 + m_dim.N0 * ( i1 + m_dim.N1 * ( i2 + m_dim.N2 * i3 ) ) << "\t";

                    int hilbert = hilbert_map.map( i0 + m_dim.N0 * i1 );

                    // std::cout << "Hilbert Index: " << ( m_dim.N0 * m_dim.N1 ) * ( i2 + m_dim.N2 * i3 ) + hilbert << "\t" << "Subflag: " << sub_flag << "\t";

                    return ( m_dim.N0 * m_dim.N1 ) * ( i2 + m_dim.N2 * i3 ) + hilbert;
                }
            };

            // rank 5
            template <typename I0, typename I1, typename I2, typename I3, typename I4>
            KOKKOS_INLINE_FUNCTION 
            constexpr size_type operator()( I0 const& i0, I1 const& i1, I2 const& i2, I3 const& i3, I4 const& i4 ) const {
                if ( sub_flag ) return i4 + m_dim.N4 * ( i3 + m_dim.N3 * ( i2 + m_dim.N2 * ( i1 + m_dim.N1 * ( i0 ) ) ) );
                else {
                    // std::cout << i0 + m_dim.N0 * ( i1 + m_dim.N1 * ( i2 + m_dim.N2 * i3 ) ) << "\t";

                    int hilbert = hilbert_map.map( i0 + m_dim.N0 * i1 );

                    // std::cout << "Hilbert Index: " << ( m_dim.N0 * m_dim.N1 ) * ( i2 + m_dim.N2 * ( i3 + m_dim.N3 * i4 ) ) + hilbert << "\t";

                    return ( m_dim.N0 * m_dim.N1 ) * ( i2 + m_dim.N2 * ( i3 + m_dim.N3 * i4 ) ) + hilbert;
                }
            };

            // rank 6
            template <typename I0, typename I1, typename I2, typename I3, typename I4, typename I5>
            KOKKOS_INLINE_FUNCTION 
            constexpr size_type operator()( I0 const& i0, I1 const& i1, I2 const& i2, I3 const& i3, I4 const& i4, I5 const& i5 ) const {
                if ( sub_flag ) return i5 + m_dim.N5 * ( i4 + m_dim.N4 * ( i3 + m_dim.N3 * ( i2 + m_dim.N2 * ( i1 + m_dim.N1 * ( i0 ) ) ) ) );
                else {
                    // std::cout << i0 + m_dim.N0 * ( i1 + m_dim.N1 * ( i2 + m_dim.N2 * i3 ) ) << "\t";

                    int hilbert = hilbert_map.map( i0 + m_dim.N0 * i1 );

                    // std::cout << "Hilbert Index: " << ( m_dim.N0 * m_dim.N1 ) * ( i2 + m_dim.N2 * ( i3 + m_dim.N3 * ( i4 + m_dim.N4 * i5 ) ) ) + hilbert << "\t";

                    return ( m_dim.N0 * m_dim.N1 ) * ( i2 + m_dim.N2 * ( i3 + m_dim.N3 * ( i4 + m_dim.N4 * i5 ) ) ) + hilbert;
                }
            };

            // rank 7
            template <typename I0, typename I1, typename I2, typename I3, typename I4, typename I5, typename I6>
            KOKKOS_INLINE_FUNCTION 
            constexpr size_type operator()( I0 const& i0, I1 const& i1, I2 const& i2, I3 const& i3, I4 const& i4, I5 const& i5, I6 const& i6 ) const {
                if ( sub_flag ) return i6 + m_dim.N6 * ( i5 + m_dim.N5 * ( i4 + m_dim.N4 * ( i3 + m_dim.N3 * ( i2 + m_dim.N2 * ( i1 + m_dim.N1 * ( i0 ) ) ) ) ) );
                else {
                    // std::cout << i0 + m_dim.N0 * ( i1 + m_dim.N1 * ( i2 + m_dim.N2 * i3 ) ) << "\t";

                    int hilbert = hilbert_map.map( i0 + m_dim.N0 * i1 );

                    // std::cout << "Hilbert Index: " << ( m_dim.N0 * m_dim.N1 ) * ( i2 + m_dim.N2 * ( i3 + m_dim.N3 * ( i4 + m_dim.N4 * ( i5 + m_dim.N5 * i6 ) ) ) ) + hilbert << "\t";

                    return ( m_dim.N0 * m_dim.N1 ) * ( i2 + m_dim.N2 * ( i3 + m_dim.N3 * ( i4 + m_dim.N4 * ( i5 + m_dim.N5 * i6 ) ) ) ) + hilbert;
                }
            };

            // rank 8
            template <typename I0, typename I1, typename I2, typename I3, typename I4, typename I5, typename I6, typename I7>
            KOKKOS_INLINE_FUNCTION 
            constexpr size_type operator()( I0 const& i0, I1 const& i1, I2 const& i2, I3 const& i3, I4 const& i4, I5 const& i5, I6 const& i6, I7 const& i7 ) const {
                if ( sub_flag ) return i7 + m_dim.N7 * ( i6 + m_dim.N6 * ( i5 + m_dim.N5 * ( i4 + m_dim.N4 * ( i3 + m_dim.N3 * ( i2 + m_dim.N2 * ( i1 + m_dim.N1 * ( i0 ) ) ) ) ) ) );
                else {
                    // std::cout << i0 + m_dim.N0 * ( i1 + m_dim.N1 * ( i2 + m_dim.N2 * i3 ) ) << "\t";

                    int hilbert = hilbert_map.map( i0 + m_dim.N0 * i1 );

                    // std::cout << "Hilbert Index: " << ( m_dim.N0 * m_dim.N1 ) * ( i2 + m_dim.N2 * ( i3 + m_dim.N3 * ( i4 + m_dim.N4 * ( i5 + m_dim.N5 * ( i6 + m_dim.N6 * i7 ) ) ) ) ) + hilbert << "\t";

                    return ( m_dim.N0 * m_dim.N1 ) * ( i2 + m_dim.N2 * ( i3 + m_dim.N3 * ( i4 + m_dim.N4 * ( i5 + m_dim.N5 * ( i6 + m_dim.N6 * i7 ) ) ) ) ) + hilbert;
                }
            };

            KOKKOS_INLINE_FUNCTION
            constexpr array_layout layout() const {
                return array_layout( m_dim.N0, m_dim.N1, m_dim.N2, m_dim.N3, m_dim.N4, m_dim.N5, m_dim.N6, m_dim.N7 );
            };

            KOKKOS_INLINE_FUNCTION constexpr size_type dimension_0() const {
                return m_dim.N0;
            };

            KOKKOS_INLINE_FUNCTION constexpr size_type dimension_1() const {
                return m_dim.N1;
            };

            KOKKOS_INLINE_FUNCTION constexpr size_type dimension_2() const {
                return m_dim.N2;
            };

            KOKKOS_INLINE_FUNCTION constexpr size_type dimension_3() const {
                return m_dim.N3;
            };

            KOKKOS_INLINE_FUNCTION constexpr size_type dimension_4() const {
                return m_dim.N4;
            };

            KOKKOS_INLINE_FUNCTION constexpr size_type dimension_5() const {
                return m_dim.N5;
            };

            KOKKOS_INLINE_FUNCTION constexpr size_type dimension_6() const {
                return m_dim.N6;
            };

            KOKKOS_INLINE_FUNCTION constexpr size_type dimension_7() const {
                return m_dim.N7;
            };

            /* Cardinality of the domain index space */
            KOKKOS_INLINE_FUNCTION
            constexpr size_type size() const {
                return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 *
                    m_dim.N6 * m_dim.N7;
            };

            /* Span of the range space */
            KOKKOS_INLINE_FUNCTION
            constexpr size_type span() const {
                return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 *
                    m_dim.N6 * m_dim.N7;
            };

            KOKKOS_INLINE_FUNCTION constexpr bool span_is_contiguous() const {
                return true;
            };

            /* Strides of dimensions */
            KOKKOS_INLINE_FUNCTION 
            constexpr size_type stride_0() const { 
                return 1; 
            };

            KOKKOS_INLINE_FUNCTION 
            constexpr size_type stride_1() const {
                return m_dim.N0;
            };

            KOKKOS_INLINE_FUNCTION 
            constexpr size_type stride_2() const {
                return m_dim.N0 * m_dim.N1;
            };

            KOKKOS_INLINE_FUNCTION 
            constexpr size_type stride_3() const {
                return m_dim.N0 * m_dim.N1 * m_dim.N2;
            };

            KOKKOS_INLINE_FUNCTION 
            constexpr size_type stride_4() const {
                return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3;
            };

            KOKKOS_INLINE_FUNCTION 
            constexpr size_type stride_5() const {
                return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4;
            };

            KOKKOS_INLINE_FUNCTION 
            constexpr size_type stride_6() const {
                return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5;
            };

            KOKKOS_INLINE_FUNCTION 
            constexpr size_type stride_7() const {
                return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 * m_dim.N6;
            };

            // Stride with [ rank ] value is the total length
            template <typename iType>
            KOKKOS_INLINE_FUNCTION void stride(iType* const s) const {
                s[0] = 1;
                if (0 < dimension_type::rank) {
                    s[1] = m_dim.N0;
                }
                if ( 1 < dimension_type::rank ) {
                    s[2] = s[1] * m_dim.N1;
                }
                if ( 2 < dimension_type::rank ) {
                    s[3] = s[2] * m_dim.N2;
                }
                if ( 3 < dimension_type::rank ) {
                    s[4] = s[3] * m_dim.N3;
                }
                if ( 4 < dimension_type::rank ) {
                    s[5] = s[4] * m_dim.N4;
                }
                if ( 5 < dimension_type::rank ) {
                    s[6] = s[5] * m_dim.N5;
                }
                if ( 6 < dimension_type::rank ) {
                    s[7] = s[6] * m_dim.N6;
                }
                if ( 7 < dimension_type::rank ) {
                    s[8] = s[7] * m_dim.N7;
                }
            };

            ViewOffset()                  = default;
            ViewOffset( const ViewOffset& ) = default;
            ViewOffset& operator=( const ViewOffset& ) = default;

            KOKKOS_INLINE_FUNCTION 
            constexpr ViewOffset( std::integral_constant<unsigned, 0> const&, Kokkos::LayoutHilbert const& rhs)
            : m_dim( rhs.dimension[0], rhs.dimension[1], rhs.dimension[2], rhs.dimension[3], rhs.dimension[4], rhs.dimension[5], rhs.dimension[6], rhs.dimension[7] ) {};

            template <class DimRHS, class LayoutRHS>
            KOKKOS_INLINE_FUNCTION 
            constexpr ViewOffset( const ViewOffset<DimRHS, LayoutRHS, void>& rhs )
            : m_dim( rhs.m_dim.N0, rhs.m_dim.N1, rhs.m_dim.N2, rhs.m_dim.N3, rhs.m_dim.N4, rhs.m_dim.N5, rhs.m_dim.N6, rhs.m_dim.N7 ) {
                static_assert( int( DimRHS::rank ) == int( dimension_type::rank ), "ViewOffset assignment requires equal rank" );
            };

            template <class DimRHS, class LayoutRHS>
            KOKKOS_INLINE_FUNCTION 
            constexpr ViewOffset( const ViewOffset<DimRHS, LayoutRHS, void>& rhs, const SubviewExtents<DimRHS::rank, dimension_type::rank>& sub )
            : m_dim( sub.range_extent(0), sub.range_extent( 1 ), sub.range_extent( 2 ), sub.range_extent( 3 ), sub.range_extent( 4 ), sub.range_extent( 5 ), sub.range_extent( 6 ), sub.range_extent( 7 ) ), sub_flag( 1 )
            {
                // std::cout << "Trace\n";
            };
        };

        template <int RankDest, int RankSrc, int CurrentArg, class Arg,
                class... SubViewArgs>
        struct SubviewLegalArgsCompileTime<Kokkos::LayoutHilbert, Kokkos::LayoutHilbert,
                                        RankDest, RankSrc, CurrentArg, Arg,
                                        SubViewArgs...> {
        enum {
            value = (((CurrentArg == RankDest - 1) &&
                    (Kokkos::Impl::is_integral_extent_type<Arg>::value)) ||
                    ((CurrentArg >= RankDest) && (std::is_integral<Arg>::value)) ||
                    ((CurrentArg < RankDest) &&
                    (std::is_same<Arg, Kokkos::Impl::ALL_t>::value)) ||
                    ((CurrentArg == 0) &&
                    (Kokkos::Impl::is_integral_extent_type<Arg>::value))) &&
                    (SubviewLegalArgsCompileTime<Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                                RankDest, RankSrc, CurrentArg + 1,
                                                SubViewArgs...>::value)
        };
        };

        template <int RankDest, int RankSrc, int CurrentArg, class Arg>
        struct SubviewLegalArgsCompileTime<Kokkos::LayoutHilbert, Kokkos::LayoutHilbert,
                                        RankDest, RankSrc, CurrentArg, Arg> {
        enum {
            value = ((CurrentArg == RankDest - 1) || (std::is_integral<Arg>::value)) &&
                    (CurrentArg == RankSrc - 1)
        };
        };
        template <class SrcTraits, class... Args>
        struct ViewMapping<
            typename std::enable_if<( std::is_same<typename SrcTraits::specialize, void>::value && ( std::is_same<typename SrcTraits::array_layout, Kokkos::LayoutHilbert>::value ) )>::type, SrcTraits, Args...> {
        
            private:
                static_assert( SrcTraits::rank == sizeof...( Args ), "Subview mapping requires one argument for each dimension of source View" );

                enum {
                    RZ = false,
                    R0 = bool( is_integral_extent<0, Args...>::value ),
                    R1 = bool( is_integral_extent<1, Args...>::value ),
                    R2 = bool( is_integral_extent<2, Args...>::value ),
                    R3 = bool( is_integral_extent<3, Args...>::value ),
                    R4 = bool( is_integral_extent<4, Args...>::value ),
                    R5 = bool( is_integral_extent<5, Args...>::value ),
                    R6 = bool( is_integral_extent<6, Args...>::value ),
                    R7 = bool( is_integral_extent<7, Args...>::value )
                };

                enum {
                    rank = unsigned( R0 ) + unsigned( R1 ) + unsigned( R2 ) + unsigned( R3 ) + unsigned( R4 ) + unsigned( R5 ) + unsigned( R6 ) + unsigned( R7 )
                };

                // Whether right-most rank is a range.
                enum { 
                    R0_rev = ( 0 == SrcTraits::rank ? RZ : ( 1 == SrcTraits::rank ? R0 : ( 2 == SrcTraits::rank ? R1 : ( 3 == SrcTraits::rank ? R2 : ( 4 == SrcTraits::rank ? R3
                        : ( 5 == SrcTraits::rank ? R4 : ( 6 == SrcTraits::rank ? R5 : ( 7 == SrcTraits::rank ? R6 : R7 ) ) ) ) ) ) ) )
                };

                // Subview's layout
                typedef typename std::conditional<( ( rank == 0 ) || 
                SubviewLegalArgsCompileTime<typename SrcTraits::array_layout, typename SrcTraits::array_layout, rank, SrcTraits::rank, 0, Args...>::value ||
                ( rank <= 2 && R0 && std::is_same<typename SrcTraits::array_layout, Kokkos::LayoutHilbert>::value ) ), typename SrcTraits::array_layout, Kokkos::LayoutHilbert>::type array_layout;

                typedef typename SrcTraits::value_type value_type;

                using data_type = typename SubViewDataType<value_type, typename Kokkos::Impl::ParseViewExtents<typename SrcTraits::data_type>::type, Args...>::type;

            public:
                typedef Kokkos::ViewTraits<data_type, array_layout, typename SrcTraits::device_type, typename SrcTraits::memory_traits> traits_type;

                typedef Kokkos::View<data_type, array_layout, typename SrcTraits::device_type, typename SrcTraits::memory_traits> type;

                template <class MemoryTraits>
                struct apply {
                    static_assert(Kokkos::Impl::is_memory_traits<MemoryTraits>::value, "");

                    typedef Kokkos::ViewTraits<data_type, array_layout, typename SrcTraits::device_type, MemoryTraits> traits_type;

                    typedef Kokkos::View<data_type, array_layout, typename SrcTraits::device_type, MemoryTraits> type;
                };

                // The presumed type is 'ViewMapping< traits_type , void >'
                // However, a compatible ViewMapping is acceptable.
                template <class DstTraits>
                KOKKOS_INLINE_FUNCTION static void assign(
                    ViewMapping<DstTraits, void>& dst,
                    ViewMapping<SrcTraits, void> const& src, Args... args) {
                    static_assert(ViewMapping<DstTraits, traits_type, void>::is_assignable, "Subview destination type must be compatible with subview " "derived type");

                    typedef ViewMapping<DstTraits, void> DstType;

                    typedef typename DstType::offset_type dst_offset_type;

                    const SubviewExtents<SrcTraits::rank, rank> extents(src.m_impl_offset.m_dim, args...);

                    dst.m_impl_offset = dst_offset_type(src.m_impl_offset, extents);


                    dst.m_impl_handle = ViewDataHandle<DstTraits>::assign(
                        src.m_impl_handle,
                        src.m_impl_offset(extents.domain_offset(0), extents.domain_offset(1),
                                        extents.domain_offset(2), extents.domain_offset(3),
                                        extents.domain_offset(4), extents.domain_offset(5),
                                        extents.domain_offset(6), extents.domain_offset(7)));
                }
        };
    }
}

#endif