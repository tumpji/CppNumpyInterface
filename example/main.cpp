#include <iostream>
#include <vector>
#include <tuple>
#include <cstdint>
#include <cassert>
#include <cmath>

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION


#include "ndarray.h"
#include "tuple.h"
#include "ndarray.h"

#include <python3.6m/Python.h>

extern "C" {

static PyObject * pass_through(PyObject *self, PyObject *args) {
    /* parse arguments */
    PyObject *data;
    if (!PyArg_ParseTuple(args, "O", &data)) {
        std::cerr << "Cannot read inptut" << std::endl;
        return NULL;
    }
    (void)self;

    //using ArrayType = int32_t;
    //constexpr unsigned ArrayDimension = 2;

    try {
        // do not cast
        NpyArray<uint64_t, 2> input_array(data, false);
        return input_array.pass_to_python();
    } catch(std::runtime_error e) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    return NULL;
}

static PyObject * set_2_4_to_1(PyObject *self, PyObject *args) {
    /* parse arguments */
    PyObject *data;
    if (!PyArg_ParseTuple(args, "O", &data)) {
        std::cerr << "Cannot read inptut" << std::endl;
        return NULL;
    }
    (void)self;


    NpyArray<float, 2> input_array(data);
    input_array.safe_get(2,4) = 1.f;
    return input_array.pass_to_python();
}

static PyObject * return_object_dimension(PyObject *self, PyObject *args) {
    /* parse arguments */
    PyObject *data;
    if (!PyArg_ParseTuple(args, "O", &data)) {
        std::cerr << "Cannot read inptut" << std::endl;
        return NULL;
    }
    (void)self;

    using ArrayType = float;
    constexpr unsigned ArrayDimension = 2;

    NpyArray<ArrayType, ArrayDimension> input_array(data);
    NpyArray<int, 1> output_array(INIT::EMPTY, 2);
    output_array.safe_get(0) = input_array.dim_sizes[0];
    output_array.safe_get(1) = input_array.dim_sizes[1];

    return output_array.pass_to_python();
}

static PyObject * insitu_cosine(PyObject *self, PyObject *args) {
    /* parse arguments */
    PyObject *data;
    if (!PyArg_ParseTuple(args, "O", &data)) {
        std::cerr << "Cannot read inptut" << std::endl;
        return NULL;
    }
    (void)self;

    using ArrayType = float;
    constexpr unsigned ArrayDimension = 2;

    NpyArray<ArrayType, ArrayDimension> input_array(data);
    for (unsigned y = 0; y < input_array.dim_sizes[0]; ++y)
        for (unsigned x = 0; x < input_array.dim_sizes[1]; ++x)
        {
            float& element = input_array.safe_get(y,x);
            element = std::cos(element);
        }

    return input_array.pass_to_python();
}

static PyObject * get_2_times_5_matrix(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    NpyArray<double, 2> a(INIT::ZERO, 2,5);
    return a.pass_to_python();
}
 

static PyObject * return_tuple(PyObject *self, PyObject *args) {
    int input;
    if (!PyArg_ParseTuple(args, "i", &input)) {
        std::cerr << "Cannot read inptut" << std::endl;
        return NULL;
    }
    (void)self;

    char string[] = "hello";
    PyTuple output_tuple(input, 1.5f, string);
    return output_tuple.pass_to_python();
}

static const char* doc = NULL;

static PyMethodDef Methods[] = {
    {"pass_through", pass_through, METH_VARARGS, "doc"},
    {"set_2_4_to_1", set_2_4_to_1, METH_VARARGS, "doc"},
    {"return_object_dimension", return_object_dimension, METH_VARARGS, "doc"},
    {"insitu_cosine", insitu_cosine, METH_VARARGS, "doc"},
    {"return_tuple", return_tuple, METH_VARARGS, "doc"},
    {"get_2_times_5_matrix", get_2_times_5_matrix, METH_VARARGS, "doc"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "example",   /* name of module */
    doc, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    Methods
};


PyMODINIT_FUNC PyInit_libexample(void)
{
    PyObject *m;

    m = PyModule_Create(&module);
    if (m == NULL)
        return NULL;

    import_array();
    if (PyErr_Occurred()) return NULL;
    /* you can create some objects here <errors...> */
    return m;
}



} // extern "C"




