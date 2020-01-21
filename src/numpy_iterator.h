#include <iostream>
#include <cstdarg>

extern "C" {
#include <site-packages/numpy/core/include/numpy/ndarrayobject.h>
#include <site-packages/numpy/core/include/numpy/ndarraytypes.h>
#include <site-packages/numpy/core/include/numpy/utils.h>
#include <site-packages/numpy/core/include/numpy/arrayobject.h>
#include <site-packages/numpy/core/include/numpy/noprefix.h>
}


template<class INTER_SAVE>
class NpyArrayIterator {
public:
    NpyArrayIterator(PyObject* obj): obj(obj), done(false)
    {
        
        init_array_properties();

        // create iterator
        iter = NpyIter_New(
                (PyArrayObject*)obj, 
                NPY_ITER_READONLY | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK, 
                NPY_CORDER, 
///NPY_KEEPORDER, 
                NPY_NO_CASTING, 
                NULL);
        if (iter == NULL) {
            throw std::runtime_error("Cannot create NPY iterator");
        }

        // create iterator next function
        iterfunc = NpyIter_GetIterNext(iter, NULL);
        if (iterfunc == NULL) {
            NpyIter_Deallocate(iter);
            throw std::runtime_error("Cannot create NPY iterator function");
        }

        // get attributes for faster reading
        outer_stats.act_dataptr = NpyIter_GetDataPtrArray(iter);
        outer_stats.act_strideptr = NpyIter_GetInnerStrideArray(iter);
        outer_stats.act_innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

        inner_stats.data = *outer_stats.act_dataptr;
        inner_stats.stride = *outer_stats.act_strideptr;
        inner_stats.count = *outer_stats.act_innersizeptr;

        // if array is empty
        if (inner_stats.count == 0) {
            done = true;
        }
    }


    INTER_SAVE& readnext();
    inline bool is_done() const {return done;}
protected:
    PyObject* obj; //python object
    NpyIter* iter; // python iterator object
    NpyIter_IterNextFunc* iterfunc; // python iteration function
    bool done; // is iterator at the end of given array

    struct {
        char** act_dataptr;
        npy_intp *act_strideptr;
        npy_intp *act_innersizeptr;
    }outer_stats;

    struct { // inner 
        char* data;
        npy_intp count;
        npy_intp stride;
    }inner_stats;

    struct { // structure contains properties of the array
        int ndims;
        int totsize;
        std::vector<int> dims;
    } array_properties;

public:
    int no_cols() {return array_properties.dims[0];}
    int no_rows() {return array_properties.dims[1];}
    void init_array_properties();
};

template<class INTER_SAVE>
INTER_SAVE& NpyArrayIterator<INTER_SAVE>::readnext() {
    INTER_SAVE& value = *reinterpret_cast<INTER_SAVE*>(inner_stats.data);

    inner_stats.count -= 1;
    inner_stats.data += inner_stats.stride;

    if (inner_stats.count == 0) {
        if(not iterfunc(iter)) {
            done = true;
            inner_stats.data = NULL;
            inner_stats.count = 0;
            inner_stats.stride = 0;
            return value;
        }

        inner_stats.data = *outer_stats.act_dataptr;
        inner_stats.count = *outer_stats.act_innersizeptr;
        inner_stats.stride = *outer_stats.act_strideptr;
    }
    return value;
}

template<class INTER_SAVE>
void NpyArrayIterator<INTER_SAVE>::init_array_properties() 
{
    this->array_properties.ndims = PyArray_NDIM(reinterpret_cast<PyArrayObject*>(obj));
    this->array_properties.totsize = PyArray_SIZE(reinterpret_cast<PyArrayObject*>(obj));
    for (int i = 0; i < array_properties.ndims; ++i){
        this->array_properties.dims.push_back(PyArray_DIM(reinterpret_cast<PyArrayObject*>(obj), i));
    }

    if(array_properties.totsize == 0) {
        this->done = true; 
    }
}


template<class INTER_SAVE>
class UpperDiagonalArrayIterator: public NpyArrayIterator<INTER_SAVE> {
    public:
    UpperDiagonalArrayIterator(PyObject* obj, bool check_square = true): 
        NpyArrayIterator<INTER_SAVE>(obj), row(1), col(0)
    { 
        NpyArrayIterator<INTER_SAVE>::readnext();
        if (check_square and NpyArrayIterator<INTER_SAVE>::array_properties.dims.size() != 2) {
            throw std::runtime_error("Numpy array is not square matrix");
        }
    }

    std::tuple<int, int, INTER_SAVE> readnext() {
        INTER_SAVE next_val;
        int next_row, next_col;

        do {
            next_val = NpyArrayIterator<INTER_SAVE>::readnext();
            next_row = row;
            next_col = col;

            row += 1;
            if (row == NpyArrayIterator<INTER_SAVE>::array_properties.dims[1]) {
                row = 0;
                col += 1;
                // todo skip
            }
            if (col >= NpyArrayIterator<INTER_SAVE>::array_properties.dims[1] - 1) {
                NpyArrayIterator<INTER_SAVE>::done = true;
            }
        } while(next_row <= next_col);

        return std::make_tuple(next_col, next_row, next_val);
    }

    inline bool is_done() const {return NpyArrayIterator<INTER_SAVE>::done;}
private:
    int row, col;
};

template<class INTER_SAVE>
class FullArrayIterator: public NpyArrayIterator<INTER_SAVE> {
public:
        FullArrayIterator(PyObject* obj):
            NpyArrayIterator<INTER_SAVE>(obj), row(0), col(0)
    {}

    std::tuple<int, int, INTER_SAVE> readnext() {
        INTER_SAVE next_val = NpyArrayIterator<INTER_SAVE>::readnext();

        std::tuple<int, int, INTER_SAVE> res = std::make_tuple(col, row, next_val);

        row += 1; 
        if (row == NpyArrayIterator<INTER_SAVE>::array_properties.dims[1]) {
            row = 0;
            col += 1;
        }
        return res;
    }

protected:
    int row, col;
};

template<class INTER_SAVE>
class FlattenedArrayIterator: public NpyArrayIterator<INTER_SAVE> {
    public:
    FlattenedArrayIterator(PyObject* obj): 
        NpyArrayIterator<INTER_SAVE>(obj), number(0)
    {}

    std::tuple<int, INTER_SAVE> readnext() {
        INTER_SAVE next_val = NpyArrayIterator<INTER_SAVE>::readnext();;
        return std::make_tuple(number++, next_val);
    }

    inline bool is_done() const {return NpyArrayIterator<INTER_SAVE>::done;}
private:
    int number;
};


