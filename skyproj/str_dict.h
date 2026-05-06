#ifndef STR_DICT_H
#define STR_DICT_H

#include <Python.h>
#include <stddef.h>

typedef struct {
    char *key;
    double value;
} DictEntry;

typedef struct {
    DictEntry *entries;
    size_t size;
    size_t capacity;
} StrDict;

// Core dictionary functions
StrDict* str_dict_create(size_t initial_capacity);
void str_dict_free(StrDict *dict);
int str_dict_set(StrDict *dict, const char *key, double value);
int str_dict_get(StrDict *dict, const char *key, double *value);
int str_dict_contains(StrDict *dict, const char *key);
void str_dict_print(StrDict *dict);

// Python integration
StrDict* str_dict_from_pydict(PyObject *py_dict);

#endif
