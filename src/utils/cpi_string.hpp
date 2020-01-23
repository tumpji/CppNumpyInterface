#if not defined CPI_UTILS_STRING_INCLUDED
#define CPI_UTILS_STRING_INCLUDED
// this code is inspired by blog:
// https://akrzemi1.wordpress.com/2017/06/28/compile-time-string-concatenation/

#include <utility>
#include "cpi_assert.hpp"

// ***********************
// literal

template <int N>
class StaticString {
public:
    char _array[N+1];

private:
    template<int N1, int... PACK1, int...PACK2>
    constexpr StaticString(
            const StaticString<N1>& s1,
            const StaticString<N - N1>& s2,
            std::integer_sequence<int, PACK1...>,
            std::integer_sequence<int, PACK2...>):
        _array{s1._array[PACK1]..., s2._array[PACK2]..., '\0'}
    {}

public:
    // from literal 
    template<int... PACK>
    constexpr StaticString(
            const char (&string)[N+1],
            std::integer_sequence<int, PACK...>):
        _array{string[PACK]..., '\0'}
    {}
    /*
    constexpr StaticString(const char (&string)[N+1] ):
        StaticString(string, std::make_integer_sequence<int,N>{})
    {}
    */

    // concatenation of two items
    template<int N1, std::enable_if_t<N1 <= N, int> = 0>
    constexpr StaticString(
            const StaticString<N1>& s1,
            const StaticString<N - N1>& s2)
    : StaticString(
            s1,
            s2,
            std::make_integer_sequence<int,N1>{}, 
            std::make_integer_sequence<int,N-N1>{}) {}


    // other funcs
    constexpr char operator[] (int i) {
        static_assert(i >= 0 and i < N+1);
        return _array[i];
    }

    constexpr std::size_t size() const { return N; }

    constexpr const char * cstring() const {return _array;}
};

template<int N1, int N2>
constexpr StaticString<N1+N2> operator+ (
        const StaticString<N1>& s1,
        const StaticString<N2>& s2)
{
    return StaticString<N1+N2>(s1,s2);
}



template <int NPO>
constexpr auto staticLiteral(const char (&string)[NPO]) -> StaticString<NPO-1>
{
    return StaticString<NPO-1>(string, std::make_integer_sequence<int,NPO-1>{});
}
template <int NPO>
constexpr auto staticLiteral(const char (&&string)[NPO]) -> StaticString<NPO-1>
{
    return StaticString<NPO-1>(string, std::make_integer_sequence<int,NPO-1>{});
}



#endif // CPI_UTILS_STRING_INCLUDED
