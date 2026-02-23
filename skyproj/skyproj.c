#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>
#include <math.h>
#include <proj.h>
#include <stdbool.h>
#include <stdio.h>

#include "skyproj.h"

PyDoc_STRVAR(transform_doc,
             "transform(projstr, a, b, degrees=True)\n"
             "--\n\n"
             "Transform from lon, lat to x, y or inverse.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "proj_str : `str`\n"
             "    proj projection string.\n"
             "a : `np.ndarray` (N,)\n"
             "    Longitude array or x array.\n"
             "b : `np.ndarray` (N,)\n"
             "    Latitude array or y array.\n"
             "degrees : `bool`, optional\n"
             "    Input/output in degrees?\n"
             "inverse : `bool`, optional\n"
             "    Do inverse transform?\n"
             "\n"
             "Returns\n"
             "-------\n"
             "xy or lonlat : `np.ndarray` (N, 2)\n"
             "    xy or lonlat array.\n");

static PyObject *transform(PyObject *dummy, PyObject *args, PyObject *kwargs) {
    PyObject *a_obj = NULL, *b_obj = NULL;
    PyObject *a_arr = NULL, *b_arr = NULL;
    PyObject *a2b2_arr = NULL;
    const char *proj_str_in = NULL;
    char proj_str[PROJ_STR_SIZE];
    int errno;

    int degrees = 1;
    int inverse = 0;

    NpyIter *iter = NULL;

    static char *kwlist[] = {"proj_str", "a", "b", "degrees", "inverse", NULL};

    double *a2b2s = NULL;
    char err[ERR_SIZE];
    bool loop_failed = false;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sOO|pp", kwlist, &proj_str_in, &a_obj,
                                     &b_obj, &degrees, &inverse))
        goto fail;

    a_arr = PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (a_arr == NULL) goto fail;
    b_arr = PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (b_arr == NULL) goto fail;

    // The input arrays are lon_arr (double), lat_arr (double).
    // The output arrays are x_arr (double), y_arr (double).

    PyArrayObject *op[2];
    npy_uint32 op_flags[2];
    PyArray_Descr *op_dtypes[2];

    op[0] = (PyArrayObject *)a_arr;
    op_flags[0] = NPY_ITER_READONLY;
    op_dtypes[0] = NULL;
    op[1] = (PyArrayObject *)b_arr;
    op_flags[1] = NPY_ITER_READONLY;
    op_dtypes[1] = NULL;

    npy_uint32 iter_flags = NPY_ITER_ZEROSIZE_OK | NPY_ITER_RANGED | NPY_ITER_BUFFERED;

    iter = NpyIter_MultiNew(2, op, iter_flags, NPY_KEEPORDER, NPY_NO_CASTING, op_flags,
                            op_dtypes);

    if (iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "lon, lat arrays could not be broadcast together.");
        goto fail;
    }

    npy_intp iter_size = NpyIter_GetIterSize(iter);
    int ndims = NpyIter_GetNDim(iter);

    // Allocate output arrays
    if (ndims == 0) {
        npy_intp dims[1];
        dims[0] = 2;
        a2b2_arr = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
        if (a2b2_arr == NULL) goto fail;
    } else {
        npy_intp dims[2];
        dims[0] = iter_size;
        dims[1] = 2;
        a2b2_arr = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
        if (a2b2_arr == NULL) goto fail;
    }
    a2b2s = (double *)PyArray_DATA((PyArrayObject *)a2b2_arr);

    if (iter_size == 0) {
        goto cleanup;
    }

    NpyIter_IterNextFunc *iternext;
    char **dataptrarray;
    char *errmsg;

    if (NpyIter_ResetToIterIndexRange(iter, 0, iter_size, &errmsg) != NPY_SUCCEED) {
        PyErr_SetString(PyExc_RuntimeError, errmsg);
        goto fail;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to get iterator next function.");
        goto fail;
    }

    snprintf(proj_str, PROJ_STR_SIZE, "%s +ellps=sphere +R=1.0 ALLOW_BALLPARK=True", proj_str_in);

    PJ *p;
    PJ_CONTEXT *c;
    PJ_COORD coord = {{0, 0, 0, 0}};
    PJ_COORD coord2 = {{0, 0, 0, 0}};

    NPY_BEGIN_ALLOW_THREADS

    c = proj_context_create();
    proj_log_level(c, PJ_LOG_NONE);
    p = proj_create(c, proj_str);
    //                "+proj=moll lon_0=0.0");
    PJ_OPERATION_FACTORY_CONTEXT *operation_ctx = proj_create_operation_factory_context(c, NULL);
    proj_operation_factory_context_set_allow_ballpark_transformations(
        c, operation_ctx, true);

    proj_errno_reset(p);

    dataptrarray = NpyIter_GetDataPtrArray(iter);

    do {
        size_t index = NpyIter_GetIterIndex(iter);

        if (inverse == 0) {
            // Forward transform.
            coord.v[0] = *(double *)dataptrarray[0];
            coord.v[1] = *(double *)dataptrarray[1];
            if (degrees) {
                coord.v[0] *= SP_D2R;
                coord.v[1] *= SP_D2R;
            }
            coord2 = proj_trans(p, PJ_FWD, coord);

            a2b2s[2*index] = coord2.enu.e;
            a2b2s[2*index + 1] = coord2.enu.n;
        } else {
            // Inverse transform.
            coord.enu.e = *(double *)dataptrarray[0];
            coord.enu.n = *(double *)dataptrarray[1];
            coord2 = proj_trans(p, PJ_INV, coord);

            a2b2s[2*index] = coord2.lp.lam;
            a2b2s[2*index + 1] = coord2.lp.phi;
            if (degrees) {
                a2b2s[2*index] *= SP_R2D;
                a2b2s[2*index + 1] *= SP_R2D;
            }
        }

        // We allow invalid coordinates; we just transform them to NaNs.
        if (!isfinite(a2b2s[2*index])) {
            // Set to NaN.
            a2b2s[2*index] = NAN;
            a2b2s[2*index + 1] = NAN;
        }

    } while (iternext(iter));

    /* If we want to check for errors, this is what we do.
    errno = proj_context_errno(c);
    if (errno > 0) {
        const char* errstr;
        errstr = proj_context_errno_string(c, errno);
    }
    */

    proj_destroy(p);
    proj_context_destroy(c);

    NPY_END_ALLOW_THREADS

cleanup:
    Py_DECREF(a_arr);
    Py_DECREF(b_arr);
    if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
        iter = NULL;
        goto fail;
    }

    return PyArray_Return((PyArrayObject *)a2b2_arr);

 fail:
    Py_XDECREF(a_arr);
    Py_XDECREF(b_arr);
    Py_XDECREF(a2b2_arr);
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }

    return NULL;
};

static PyMethodDef cskyproj_methods[] = {
    {"transform", (PyCFunction)(void (*)(void))transform,
     METH_VARARGS | METH_KEYWORDS, transform_doc},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef cskyproj_module = {PyModuleDef_HEAD_INIT, "_cskyproj", NULL, -1,
                                            cskyproj_methods};

PyMODINIT_FUNC PyInit__cskyproj(void) {
    import_array();
    return PyModule_Create(&cskyproj_module);
}
