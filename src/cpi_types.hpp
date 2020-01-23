#if not defined  CPI_TYPES_INCLUDED
#define  CPI_TYPES_INCLUDED

#include "ndarraytypes.h"
#include "cpi_assert.hpp"

template<typename T>
 NPY_TYPES to_numpy_type() {
     X_ASSERT(false);
     return NPY_TYPES::NPY_VOID;
 }
template<>
 NPY_TYPES to_numpy_type<bool>() {return NPY_TYPES::NPY_BOOL;}

template<>
 NPY_TYPES to_numpy_type<char>(){return NPY_TYPES::NPY_BYTE;}
template<>
 NPY_TYPES to_numpy_type<unsigned char>(){return NPY_TYPES::NPY_UBYTE;}
template<>
 NPY_TYPES to_numpy_type<short>(){return NPY_TYPES::NPY_SHORT;}
template<>
 NPY_TYPES to_numpy_type<unsigned short>(){return NPY_TYPES::NPY_USHORT;}
template<>
 NPY_TYPES to_numpy_type<int>(){return NPY_TYPES::NPY_INT;}
template<>
 NPY_TYPES to_numpy_type<unsigned int>(){return NPY_TYPES::NPY_UINT;}
template<>
 NPY_TYPES to_numpy_type<long>(){return NPY_TYPES::NPY_LONG;}
template<>
 NPY_TYPES to_numpy_type<unsigned long>(){return NPY_TYPES::NPY_ULONG;}
template<>
 NPY_TYPES to_numpy_type<long long>(){return NPY_TYPES::NPY_LONGLONG;}
template<>
 NPY_TYPES to_numpy_type<unsigned long long>(){return NPY_TYPES::NPY_ULONGLONG;}
template<>
 NPY_TYPES to_numpy_type<double>(){return NPY_TYPES::NPY_DOUBLE;}
template<>
 NPY_TYPES to_numpy_type<float>(){return NPY_TYPES::NPY_FLOAT;}


#endif // CPI_TYPES_INCLUDED
