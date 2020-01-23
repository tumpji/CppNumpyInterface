#if not defined NPY_ARRAY_INCLUDED
#define NPY_ARRAY_INCLUDED

#include <iostream>
#include <tuple>
#include <cstdint>
#include <string>

#include <exception>
#include <array>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "python3.6m/Python.h"
#include <ndarrayobject.h>

#include "cpi_types.hpp"
#include "utils/cpi_bool.hpp"
#include "utils/cpi_convert.hpp"


enum class INIT {EMPTY, ZERO};

template<class T, unsigned DIMS=1>
class NpyArray {
public:
    // CONSTRUCTORS
    // A) GENERATIVE CONSTRUCTORS
    // A.1) constructor <type1 type2, ...> with checking
    template<typename... Args, 
         std::enable_if_t<!types::all_same<unsigned, Args...>::value>* = nullptr, // not all unsigned
         std::enable_if_t<types::all_true<std::is_integral<Args>...>::value>* = nullptr // integral arguments
         >
    inline NpyArray(INIT initialization, Args... args): 
        NpyArray::NpyArray(initialization, convert::to_unsigned(args)...)
    {/* fake constructor */}

    // A.2) constructor <unsigned ...> without checking
    template<
         typename... Args, 
         std::enable_if_t<types::all_same<unsigned, Args...>::value>* = nullptr, // all unsigned
         char(*)[sizeof...(Args) == DIMS] = nullptr // number is correct
         >
    explicit NpyArray(INIT initialization, Args... args): dim_sizes({args...}) {
        create_python_object(initialization);
    }

    // B) LOAD EXISTING NUMPY OBJECT
    NpyArray(PyObject* object, bool enable_cast=false);

    // C) COPY SOMETHING WITH UNSPECIFIED ORIGIN
    NpyArray(const NpyArray& other) : NpyArray(other.object) 
    { }

    // DESTRUCTOR
    ~NpyArray();


    // provide object to python remove refs and make this object unusable
    PyObject* pass_to_python();

    /// GETTERS (use iterator if possible)
    // safe_get (checks almost everything)
    template<typename... Args, 
         std::enable_if_t<!types::all_same<unsigned, Args...>::value>* = nullptr, // all unsigned
         std::enable_if_t<types::all_true<std::is_integral<Args>...>::value>* = nullptr // integral arguments
         >
    inline T& safe_get(Args... args) { return safe_get(convert::to_unsigned(args)...); }
    template<typename... Args, 
         std::enable_if_t<types::all_same<unsigned, Args...>::value>* = nullptr, // all unsigned
         std::enable_if_t<types::all_true<std::is_integral<Args>...>::value>* = nullptr // integral arguments
         >
    T& safe_get(Args... args) 
    {
        if (check_dimension(0, args...))
            std::overflow_error("Dimension index overflow");
        return unsafe_get(args...);
    }

    template<typename... Args>
    inline bool check_dimension(unsigned index, unsigned arg, Args... args) const {
        if (arg >= dim_sizes[index])
            return true;
        return check_dimension(index+1, args...);
    }
    inline bool check_dimension(unsigned index, unsigned arg) const {
        if (arg >= dim_sizes[index])
            return true;
        return false;
    }

    // unsafe_get
    template<typename... Args, 
         std::enable_if_t<types::all_same<unsigned, Args...>::value>* = nullptr, // all unsigned
         std::enable_if_t<types::all_true<std::is_integral<Args>...>::value>* = nullptr // integral arguments
         >
    inline T& unsafe_get(Args... args) {
        npy_intp index[DIMS] = {args...};

        T* e = reinterpret_cast<T*>(PyArray_GetPtr(
                reinterpret_cast<PyArrayObject*>(object), 
                reinterpret_cast<npy_intp*>(&index)
                ));
        return *e;
    }


    std::array<unsigned, DIMS> dim_sizes;
private:
    const static NPY_TYPES numpy_dtype;
    PyObject* object;

    void create_python_object(INIT);
};


template<typename T, unsigned DIMS>
NpyArray<T,DIMS>::NpyArray(PyObject* object, bool enable_cast)
{
    this->object = object;
    // #dims
    if (PyArray_NDIM(reinterpret_cast<PyArrayObject*>(object)) != DIMS)
        throw std::runtime_error("NpyArray: input object must have the same number of dimensions");
    // save dims
    for(unsigned i = 0; i < DIMS; ++i) {
        dim_sizes[i] = PyArray_DIM(reinterpret_cast<PyArrayObject*>(object), i);
    }
    // casting
    if (PyArray_TYPE(reinterpret_cast<PyArrayObject*>(object)) != numpy_dtype) {
        if (enable_cast) {
            // try to cast 
            this->object = PyArray_CastToType(
                    reinterpret_cast<PyArrayObject*>(object), 
                    PyArray_DescrFromType(numpy_dtype), 
                    0);
            // remove reference to previous version
            Py_DECREF(object);
            object = nullptr;
        }
        else 
            throw std::runtime_error("Passed Object have different type");
    }
}

template<class T, unsigned DIMS>
NpyArray<T,DIMS>::~NpyArray() {
  //Py_DECREF(object); 
}

template<class T, unsigned DIMS>
PyObject* NpyArray<T,DIMS>::pass_to_python() {
  Py_INCREF(object); 
  return object;
}


template<class T, unsigned DIMS>
const NPY_TYPES NpyArray<T,DIMS>::numpy_dtype = to_numpy_type<T>();

template<class T, unsigned DIMS>
void NpyArray<T,DIMS>::create_python_object(INIT init) {
    (void)init;

    // 1) set up shape
    npy_intp output_shape[DIMS];
    for (unsigned i = 0; i < DIMS; ++i) 
        output_shape[i] = dim_sizes[i];

    // 2) create python object
    if (init == INIT::EMPTY) {
        object = PyArray_Empty(DIMS, output_shape, PyArray_DescrFromType(numpy_dtype), 0);
        //object = PyArray_
    } else if (init == INIT::ZERO) {
        object = PyArray_Zeros(DIMS, output_shape, PyArray_DescrFromType(numpy_dtype), 0);
    }
    else 
        throw std::runtime_error("FILL TYPE NOT IMPLEMENTED");

    if (object == nullptr)
        std::runtime_error("Cannot create numpy object");
}

#endif // NPY_ARRAY_INCLUDED
