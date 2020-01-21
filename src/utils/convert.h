#if not defined UTILS_CONVERT_INCLUDED
#define UTILS_CONVERT_INCLUDED

#include <type_traits>
#include <limits>
#include <numeric>

namespace convert {
    // function to_unsigned:
    // no additional checks are needed 
    template<
        class T, 
        std::enable_if_t<std::is_unsigned<T>::value>* = nullptr, // unsigned
        char(*)[std::numeric_limits<unsigned>::max() >= std::numeric_limits<T>::max()] = nullptr // smaller or equal range
        >
    inline constexpr unsigned to_unsigned(T x) {
        return static_cast<unsigned>(x);
    }
    // function to_unsigned:
    // checks if the entering type is signed
    // if yes it adds check to avoid overflow
    template<
        class T, 
        std::enable_if_t<std::is_signed<T>::value>* = nullptr
        >
    inline constexpr unsigned to_unsigned(T x) {
        using NEW_TYPE = typename std::make_unsigned<T>::type;
        if (x < 0)
            throw std::runtime_error("Dimensions should be only possitive values.");
        return to_unsigned(static_cast<NEW_TYPE>(x));
    }

    // function to_unsigned:
    // checks if the entering type can be greather than unsigned type
    // if yes it adds check to avoid overflow
    template<
        class T, 
        std::enable_if_t<std::is_unsigned<T>::value>* = nullptr, // unsigned
        char(*)[std::numeric_limits<unsigned>::max() < std::numeric_limits<T>::max()] = nullptr // bigger range than unsignegned int
        >
    inline constexpr unsigned to_unsigned(T x) {
        if (x > std::numeric_limits<unsigned>::max())
            throw std::runtime_error("Dimension size overflow");
        return static_cast<unsigned>(x);
    }


} // end namespace convert

#endif // UTILS_CONVERT_INCLUDED
