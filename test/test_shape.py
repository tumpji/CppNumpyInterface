import pytest 
import numpy as np
import os

import libexample


def test_pass_through():
    v = libexample.pass_through( np.zeros((11,14), dtype=np.uint64) );
    assert v.shape == (11,14)
    assert v.dtype == np.uint64

def test_pass_through_invalid_dtype():
    v = libexample.pass_through( np.zeros((41,14), dtype=np.int64) );
    assert v is None

def test_pass_through_invalid_shape():
    v = libexample.pass_through( np.zeros((41,), dtype=np.uint64) );
    assert v is None

def test_set_2_4_to_1():
    e = np.zeros((11,14), dtype=np.float32)
    new_e = libexample.set_2_4_to_1(e)
    assert new_e[2,4] == 1
    assert e[2,4] == 1
    e[2,4] = 0
    assert np.sum(e) == 0.

def test_return_object_dimension():
    e = np.empty((13,2), np.float32)
    m = libexample.return_object_dimension(e)
    print(m)

    assert m[0] == 13
    assert m[1] == 2

    e = np.empty((13,2), np.float32)


def test_in_situ_cosine():
    orig = np.float32(np.random.uniform(size=(5,4)))
    e = np.copy(orig)
    m = libexample.insitu_cosine(e)

    assert np.all(e == m)
    assert np.allclose(e,np.cos(orig))

def test_get_2_times_5_matrix():
    m = libexample.get_2_times_5_matrix()
    assert m.shape == (2,5)
    assert np.all(m == 0)
    assert np.all(m.dtype == np.float64)

def test_return_tuple():
    m = libexample.return_tuple(123)
    assert m[0] == 123
    assert m[1] == 1.5
    assert m[2] == "hello"

    a = libexample.return_tuple(1)
    assert a[0] == 1
    assert a[1] == 1.5
    assert a[2] == "hello"



#def test_false():
#    assert 5 == 4

