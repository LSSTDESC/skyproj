#ifndef _SKYPROJ_H

#define _SKYPROJ_H

#include <numpy/arrayobject.h>

#define ERR_SIZE 256
#define PROJ_STR_SIZE 1024

#define MIN_CHUNK_SIZE 10000

#define SP_PI 3.141592653589793238462643383279502884197
#define SP_D2R SP_PI / 180.0
#define SP_R2D 180.0 / SP_PI

typedef struct {
    NpyIter *iter;
    npy_intp start_idx;
    npy_intp end_idx;
    int degrees;
    int inverse;
    const char *proj_str;
    double *a2b2s;
    char err[ERR_SIZE];
    bool failed;
} TransformThreadData;


#endif
