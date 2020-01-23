#if not defined CPI_UTILS_BOOLEAN_INCLUDED
#define CPI_UTILS_BOOLEAN_INCLUDED

#include <type_traits>

namespace types 
{
    namespace detail
    {
        template<bool...> struct bool_pack;
        template<bool... bs>
            using all_true = std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...>>;
    } // end namespace detail



// author: TartanLlama
template <typename... Ts>
using all_true = detail::all_true<Ts::value...>;

// author: TartanLlama
template <typename T, typename... Ts>
using all_same = all_true<std::is_same<T,Ts>...>;

} // end namespace types


#endif // CPI_UTILS_BOOLEAN_INCLUDED
