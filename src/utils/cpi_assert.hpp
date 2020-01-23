// this code is inspired by blog:
// https://akrzemi1.wordpress.com/2017/05/18/asserts-in-constexpr-functions/

#if not defined CPI_ASSERT_INCLUDED
#define CPI_ASSERT_INCLUDED



#if defined __GNUC__
# define LIKELY(EXPR)  __builtin_expect(!!(EXPR), 1)
#else
# define LIKELY(EXPR)  (!!(EXPR))
#endif
 
#if defined NDEBUG
# define X_ASSERT(CHECK) void(0)
#else
# define X_ASSERT(CHECK) \
        ( LIKELY(CHECK) ?  void(0) : []{assert(!#CHECK);}() )
#endif




#endif // CPI_ASSERT_INCLUDED
