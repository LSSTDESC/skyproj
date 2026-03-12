#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>
#include <math.h>
#include <proj.h>
#include <geodesic.h>
#include <stdbool.h>
#include <stdio.h>

#include "thread_compat.h"
#include "skyproj.h"

PyDoc_STRVAR(transform_doc,
             "transform(projstr, a, b, degrees=True, inverse=False, n_threads=1)\n"
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
             "n_threads : `int`, optional\n"
             "    Number of threads to use.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "xy or lonlat : `np.ndarray` (N, 2)\n"
             "    xy or lonlat array.\n");

// Core processing logic for transform - used by both single and multi-threaded paths
static bool transform_iteration(NpyIter *iter, int degrees, int inverse, const char *proj_str,
                                double *a2b2s, npy_intp start_idx, npy_intp end_idx, char *err) {
    NpyIter_IterNextFunc *iternext;
    char **dataptrarray;
    char *errmsg;
    PJ *p = NULL;
    PJ_CONTEXT *c = NULL;
    PJ_OPERATION_FACTORY_CONTEXT *operation_ctx = NULL;
    PJ_COORD coord = {{0, 0, 0, 0}};
    PJ_COORD coord2 = {{0, 0, 0, 0}};

    // Create PROJ context and projection for this thread
    c = proj_context_create();
    if (c == NULL) {
        snprintf(err, ERR_SIZE, "Failed to create PROJ context");
        goto fail;
    }

    proj_log_level(c, PJ_LOG_NONE);
    p = proj_create(c, proj_str);
    if (p == NULL) {
        snprintf(err, ERR_SIZE, "Failed to create PROJ projection");
        goto fail;
    }

    operation_ctx = proj_create_operation_factory_context(c, NULL);
    if (operation_ctx == NULL) {
        snprintf(err, ERR_SIZE, "Failed to create PROJ operation context");
        goto fail;
    }
    proj_operation_factory_context_set_allow_ballpark_transformations(c, operation_ctx, true);

    proj_errno_reset(p);

    // For ranged iteration, reset to the specified range
    if (NpyIter_ResetToIterIndexRange(iter, start_idx, end_idx, &errmsg) != NPY_SUCCEED) {
        snprintf(err, ERR_SIZE, "%s", errmsg);
        goto fail;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        snprintf(err, ERR_SIZE, "Failed to get iterator next function");
        goto fail;
    }

    dataptrarray = NpyIter_GetDataPtrArray(iter);

    do {
        size_t index = NpyIter_GetIterIndex(iter);

        if (inverse == 0) {
            // Forward transform
            coord.v[0] = *(double *)dataptrarray[0];
            coord.v[1] = *(double *)dataptrarray[1];
            if (degrees) {
                coord.v[0] *= SP_D2R;
                coord.v[1] *= SP_D2R;
            }
            coord2 = proj_trans(p, PJ_FWD, coord);

            a2b2s[2 * index] = coord2.enu.e;
            a2b2s[2 * index + 1] = coord2.enu.n;
        } else {
            // Inverse transform
            coord.enu.e = *(double *)dataptrarray[0];
            coord.enu.n = *(double *)dataptrarray[1];
            coord2 = proj_trans(p, PJ_INV, coord);

            a2b2s[2 * index] = coord2.lp.lam;
            a2b2s[2 * index + 1] = coord2.lp.phi;
            if (degrees) {
                a2b2s[2 * index] *= SP_R2D;
                a2b2s[2 * index + 1] *= SP_R2D;
            }
        }

        // We allow invalid coordinates; we just transform them to NaNs
        if (!isfinite(a2b2s[2 * index])) {
            a2b2s[2 * index] = NAN;
            a2b2s[2 * index + 1] = NAN;
        }

    } while (iternext(iter));

    proj_destroy(p);
    proj_context_destroy(c);
    return true;

 fail:
    if (p != NULL) proj_destroy(p);
    if (c != NULL) proj_context_destroy(c);

    return false;
}

// Worker function for each thread
static void *transform_worker(void *arg) {
    TransformThreadData *td = (TransformThreadData *)arg;
    td->failed = !transform_iteration(td->iter, td->degrees, td->inverse, td->proj_str,
                                      td->a2b2s, td->start_idx, td->end_idx, td->err);
    return NULL;
}

static PyObject *transform(PyObject *dummy, PyObject *args, PyObject *kwargs) {
    PyObject *a_obj = NULL, *b_obj = NULL;
    PyObject *a_arr = NULL, *b_arr = NULL;
    PyObject *a2b2_arr = NULL;
    const char *proj_str_in = NULL;
    char proj_str[PROJ_STR_SIZE];

    NpyIter *iter = NULL;
    thread_handle_t *threads = NULL;
    TransformThreadData *thread_data = NULL;

    int degrees = 1;
    int inverse = 0;
    int n_threads = 1;

    static char *kwlist[] = {"proj_str", "a", "b", "degrees", "inverse", "n_threads", NULL};

    double *a2b2s = NULL;
    char err[ERR_SIZE];
    bool loop_failed = false;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sOO|ppi", kwlist, &proj_str_in, &a_obj,
                                     &b_obj, &degrees, &inverse, &n_threads))
        goto fail;

    a_arr = PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (a_arr == NULL) goto fail;
    b_arr = PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (b_arr == NULL) goto fail;

    // The input arrays are a_arr (double), b_arr (double).
    // The output array is a2b2_arr (double).

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
    if (n_threads > 1) {
        iter_flags |= NPY_ITER_DELAY_BUFALLOC | NPY_ITER_GROWINNER;
    }

    iter = NpyIter_MultiNew(2, op, iter_flags, NPY_KEEPORDER, NPY_NO_CASTING, op_flags,
                            op_dtypes);

    if (iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "a, b arrays could not be broadcast together.");
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

    // Prepare projection string
    snprintf(proj_str, PROJ_STR_SIZE, "%s +ellps=sphere ALLOW_BALLPARK=True", proj_str_in);

    if (n_threads > 1) {
        // Don't use threading if chunks would be too small
        if (iter_size / n_threads < MIN_CHUNK_SIZE) {
            n_threads = iter_size / MIN_CHUNK_SIZE;
        }
    }

    if (n_threads > 1) {
        // Multi-threaded path
        if (n_threads > iter_size) {
            n_threads = iter_size;
        }

        threads = malloc(n_threads * sizeof(thread_handle_t));
        thread_data = malloc(n_threads * sizeof(TransformThreadData));

        if (threads == NULL || thread_data == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate thread resources");
            goto fail;
        }

        // Create iterator copies (with GIL held)
        thread_data[0].iter = iter;
        for (int t = 1; t < n_threads; t++) {
            thread_data[t].iter = NpyIter_Copy(iter);
            if (thread_data[t].iter == NULL) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to copy iterator for threading");
                goto fail;
            }
        }

        // Divide work among threads
        npy_intp chunk_size = iter_size / n_threads;
        npy_intp remainder = iter_size % n_threads;
        bool thread_creation_failed = false;

        for (int t = 0; t < n_threads; t++) {
            thread_data[t].start_idx = t * chunk_size + (t < remainder ? t : remainder);
            thread_data[t].end_idx =
                thread_data[t].start_idx + chunk_size + (t < remainder ? 1 : 0);
            thread_data[t].degrees = degrees;
            thread_data[t].inverse = inverse;
            thread_data[t].proj_str = proj_str;
            thread_data[t].a2b2s = a2b2s;
            thread_data[t].failed = false;
            thread_data[t].err[0] = '\0';
        }

        NPY_BEGIN_ALLOW_THREADS

        for (int t = 0; t < n_threads; t++) {
            if (thread_create(&threads[t], transform_worker, &thread_data[t]) != 0) {
                snprintf(err, ERR_SIZE, "Failed to create thread %d", t);
                thread_creation_failed = true;
                n_threads = t;
                break;
            }
        }

        // Join all successfully created threads
        for (int t = 0; t < n_threads; t++) {
            thread_join(threads[t]);
        }

        NPY_END_ALLOW_THREADS

        if (thread_creation_failed) {
            PyErr_SetString(PyExc_RuntimeError, err);
            goto fail;
        }

        // Check for processing errors
        for (int t = 0; t < n_threads; t++) {
            if (thread_data[t].failed) {
                PyErr_SetString(PyExc_ValueError, thread_data[t].err);
                goto fail;
            }
        }

    } else {
        // Single-threaded path
        NPY_BEGIN_ALLOW_THREADS

        loop_failed = !transform_iteration(iter, degrees, inverse, proj_str, a2b2s,
                                           0, iter_size, err);

        NPY_END_ALLOW_THREADS

        if (loop_failed) {
            PyErr_SetString(PyExc_ValueError, err);
            goto fail;
        }
    }

cleanup:
    Py_DECREF(a_arr);
    Py_DECREF(b_arr);
    if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
        iter = NULL;
        goto fail;
    }
    if (thread_data != NULL) {
        for (int t = 1; t < n_threads; t++) {
            if (thread_data[t].iter != NULL) {
                NpyIter_Deallocate(thread_data[t].iter);
            }
        }
        free(thread_data);
    }
    if (threads != NULL) free(threads);

    return PyArray_Return((PyArrayObject *)a2b2_arr);

fail:
    Py_XDECREF(a_arr);
    Py_XDECREF(b_arr);
    Py_XDECREF(a2b2_arr);
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }

    if (thread_data != NULL) {
        for (int t = 1; t < n_threads; t++) {
            if (thread_data[t].iter != NULL) {
                NpyIter_Deallocate(thread_data[t].iter);
            }
        }
        free(thread_data);
    }
    if (threads != NULL) free(threads);

    return NULL;
}

PyDoc_STRVAR(geodesic_interp_doc,
             "geodesic_interp(lon0, lat0, lon1, lat1, npts, radius=1.0, flattening=0.0, include_start=False, include_end=True)\n"
             "--\n\n"
             "Compute geodesic points between two terminii.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "lon0 : `float`\n"
             "    Initial longitude (degrees).\n"
             "lat0 : `float`\n"
             "    Initial latitude (degrees).\n"
             "lon1 : `float`\n"
             "    Final longitude (degrees).\n"
             "lat1 : `float`\n"
             "    Final latitude (degrees).\n"
             "npts : `int`\n"
             "    Number of points to interpolate.\n"
             "radius : `float`\n"
             "    Radius of sphere (meters).\n"
             "flattening : `float`\n"
             "    Flattening parameter.\n"
             "include_start : `bool`\n"
             "    Include initial lon/lat in output?\n"
             "include_end : `bool`\n"
             "    Include final lon/lat in output?\n"
             "\n"
             "Returns\n"
             "-------\n"
             "lonlat : `np.ndarray` (N, 2)\n");

static PyObject *geodesic_interp(PyObject *dummy, PyObject *args, PyObject *kwargs) {
    PyObject *lonlat_arr = NULL;

    double lon0, lat0, lon1, lat1;
    int npts;
    double radius = 1.0;
    double flattening = 0.0;
    int include_start = 0;
    int include_end = 1;

    double *lonlat_data = NULL;

    struct geod_geodesic g;
    struct geod_geodesicline l;
    int index;

    static char *kwlist[] = {"lon0", "lat0", "lon1", "lat1", "npts", "radius", "flattening", "include_start", "include_end", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ddddi|ddpp", kwlist, &lon0, &lat0,
                                     &lon1, &lat1, &npts, &radius, &flattening,
                                     &include_start, &include_end))
        goto fail;

    // Check that npts is valid.

    npy_intp dims[2];
    dims[0] = npts;
    dims[1] = 2;
    lonlat_arr = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    if (lonlat_arr == NULL) goto fail;

    lonlat_data = (double *)PyArray_DATA((PyArrayObject *)lonlat_arr);

    geod_init(&g, radius, flattening);
    // Note geod_inverseline takes lat, lon order.
    geod_inverseline(&l, &g, lat0, lon0, lat1, lon1, GEOD_LATITUDE | GEOD_LONGITUDE);

    int npts_stepsize = npts;
    int offset = 0;

    if ((include_start == 0) & (include_end == 1)) {
        offset = 1;
    } else if ((include_start == 1) & (include_end == 1)) {
        npts_stepsize -= 1;
    } else if ((include_start == 0) & (include_end == 0)) {
        npts_stepsize += 1;
        offset = 1;
    }

    double stepsize = 1. / (double) npts_stepsize;

    for (index = 0; index < npts; index++) {
        // Note geod_genposition returns lat, lon order.
        geod_genposition(&l, GEOD_ARCMODE, (index + offset) * l.a13 * stepsize,
                         &lonlat_data[index * 2 + 1], &lonlat_data[index * 2],
                         0, 0, 0, 0, 0, 0);
    }

    return PyArray_Return((PyArrayObject *)lonlat_arr);

 fail:
    Py_XDECREF(lonlat_arr);

    return NULL;
}

PyDoc_STRVAR(geodesic_direct_doc,
             "geodesic_direct(lon, lat, az, dist, radius=1.0, flattening=0.0)\n"
             "--\n\n"
             "Solve the direct geodesic problem.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "lon : `np.ndarray`\n"
             "    Array of longitudes (degrees).\n"
             "lat : `np.ndarray`\n"
             "    Array of latitudes (degrees).\n"
             "az : `np.ndarray`\n"
             "    Array of azimuths (degrees).\n"
             "dist : `np.ndarray`\n"
             "    Array of distances (meters).\n"
             "radius : `float`\n"
             "    Radius of sphere (meters).\n"
             "flattening : `float`\n"
             "    Flattening parameter.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "lon_out : `np.ndarray`\n"
             "    Longitude points (degrees).\n"
             "lat_out : `np.ndarray`\n"
             "    Latitude points (degrees).\n");

static PyObject *geodesic_direct(PyObject *dummy, PyObject *args, PyObject *kwargs) {
    PyObject *lon_obj = NULL, *lat_obj = NULL, *az_obj = NULL, *dist_obj = NULL;
    PyObject *lon_arr = NULL, *lat_arr = NULL, *az_arr = NULL, *dist_arr = NULL;
    PyObject *lon_out_arr = NULL, *lat_out_arr = NULL;

    double radius = 1.0;
    double flattening = 0.0;

    NpyIter *iter = NULL;
    NpyIter_IterNextFunc *iternext;
    char **dataptrarray;

    struct geod_geodesic g;

    static char *kwlist[] = {"lon", "lat", "az", "dist", "radius", "flattening", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO|dd", kwlist, &lon_obj, &lat_obj,
                                     &az_obj, &dist_obj, &radius, &flattening))
        goto fail;

    lon_arr = PyArray_FROM_OTF(lon_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (lon_arr == NULL) goto fail;
    lat_arr = PyArray_FROM_OTF(lat_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (lat_arr == NULL) goto fail;
    az_arr = PyArray_FROM_OTF(az_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (az_arr == NULL) goto fail;
    dist_arr = PyArray_FROM_OTF(dist_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (dist_arr == NULL) goto fail;

    // The input arrays are lon_arr (double), lat_arr (double), az_arr (double),
    // and dist_arr (double).
    // The output arrays are lon_out_arr (double), lat_out_arr (double).
    PyArrayObject *op[6];
    npy_uint32 op_flags[6];
    PyArray_Descr *op_dtypes[6];

    op[0] = (PyArrayObject *)lon_arr;
    op_flags[0] = NPY_ITER_READONLY;
    op_dtypes[0] = NULL;
    op[1] = (PyArrayObject *)lat_arr;
    op_flags[1] = NPY_ITER_READONLY;
    op_dtypes[1] = NULL;
    op[2] = (PyArrayObject *)az_arr;
    op_flags[2] = NPY_ITER_READONLY;
    op_dtypes[2] = NULL;
    op[3] = (PyArrayObject *)dist_arr;
    op_flags[3] = NPY_ITER_READONLY;
    op_dtypes[3] = NULL;
    op[4] = NULL;
    op_flags[4] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;
    op_dtypes[4] = PyArray_DescrFromType(NPY_DOUBLE);
    op[5] = NULL;
    op_flags[5] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;
    op_dtypes[5] = PyArray_DescrFromType(NPY_DOUBLE);

    npy_uint32 iter_flags = NPY_ITER_ZEROSIZE_OK;

    iter = NpyIter_MultiNew(6, op, iter_flags, NPY_KEEPORDER, NPY_NO_CASTING, op_flags,
                            op_dtypes);
    if (iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "lon, lat, az, dist arrays could not be broadcast together.");
        goto fail;
    }

    npy_intp iter_size = NpyIter_GetIterSize(iter);

    if (iter_size == 0) {
        goto cleanup;
    }

    geod_init(&g, radius, flattening);

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Could not get iternext.");
        goto fail;
    }
    dataptrarray = NpyIter_GetDataPtrArray(iter);

    do {
        double *lon = (double *)dataptrarray[0];
        double *lat = (double *)dataptrarray[1];
        double *az = (double *)dataptrarray[2];
        double *dist = (double *)dataptrarray[3];
        double *lon_out = (double *)dataptrarray[4];
        double *lat_out = (double *)dataptrarray[5];

        geod_direct(&g, *lat, *lon, *az, *dist, lat_out, lon_out, 0);

    } while (iternext(iter));

 cleanup:
    lon_out_arr = (PyObject *)NpyIter_GetOperandArray(iter)[4];
    Py_INCREF(lon_out_arr);
    lat_out_arr = (PyObject *)NpyIter_GetOperandArray(iter)[5];
    Py_INCREF(lat_out_arr);

    Py_DECREF(lon_arr);
    Py_DECREF(lat_arr);
    Py_DECREF(az_arr);
    Py_DECREF(dist_arr);
    if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
        iter = NULL;
        goto fail;
    }

    PyObject *retval = PyTuple_New(2);
    PyTuple_SET_ITEM(retval, 0, PyArray_Return((PyArrayObject *)lon_out_arr));
    PyTuple_SET_ITEM(retval, 1, PyArray_Return((PyArrayObject *)lat_out_arr));

    return retval;

 fail:
    Py_XDECREF(lon_arr);
    Py_XDECREF(lat_arr);
    Py_XDECREF(az_arr);
    Py_XDECREF(dist_arr);

    return NULL;
}


static PyMethodDef cskyproj_methods[] = {
    {"transform", (PyCFunction)(void (*)(void))transform,
     METH_VARARGS | METH_KEYWORDS, transform_doc},
    {"geodesic_interp", (PyCFunction)(void (*)(void))geodesic_interp,
     METH_VARARGS | METH_KEYWORDS, geodesic_interp_doc},
    {"geodesic_direct", (PyCFunction)(void (*)(void))geodesic_direct,
     METH_VARARGS | METH_KEYWORDS, geodesic_direct_doc},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef cskyproj_module = {PyModuleDef_HEAD_INIT, "_cskyproj", NULL, -1,
                                            cskyproj_methods};

PyMODINIT_FUNC PyInit__cskyproj(void) {
    import_array();
    return PyModule_Create(&cskyproj_module);
}
