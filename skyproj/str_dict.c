#include "str_dict.h"
#include <Python.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

StrDict* str_dict_create(size_t initial_capacity) {
    StrDict *dict = malloc(sizeof(StrDict));
    if (!dict) return NULL;

    dict->entries = malloc(sizeof(DictEntry) * initial_capacity);
    if (!dict->entries) {
        free(dict);
        return NULL;
    }

    dict->size = 0;
    dict->capacity = initial_capacity;

    // Initialize all entries
    for (size_t i = 0; i < initial_capacity; i++) {
        dict->entries[i].key = NULL;
        dict->entries[i].value = 0.0;
    }

    return dict;
}

void str_dict_free(StrDict *dict) {
    if (!dict) return;

    for (size_t i = 0; i < dict->size; i++) {
        free(dict->entries[i].key);
    }
    free(dict->entries);
    free(dict);
}

static int str_dict_resize(StrDict *dict) {
    size_t new_capacity = dict->capacity * 2;
    DictEntry *new_entries = realloc(dict->entries, sizeof(DictEntry) * new_capacity);

    if (!new_entries) return -1;

    dict->entries = new_entries;
    dict->capacity = new_capacity;

    // Initialize new entries
    for (size_t i = dict->size; i < new_capacity; i++) {
        dict->entries[i].key = NULL;
        dict->entries[i].value = 0.0;
    }

    return 0;
}

int str_dict_set(StrDict *dict, const char *key, double value) {
    if (!dict || !key) return -1;

    // Check if key exists, update if so
    for (size_t i = 0; i < dict->size; i++) {
        if (strcmp(dict->entries[i].key, key) == 0) {
            dict->entries[i].value = value;
            return 0;
        }
    }

    // Add new entry
    if (dict->size >= dict->capacity) {
        if (str_dict_resize(dict) != 0) return -1;
    }

    dict->entries[dict->size].key = strdup(key);
    if (!dict->entries[dict->size].key) return -1;

    dict->entries[dict->size].value = value;
    dict->size++;

    return 0;
}

int str_dict_get(StrDict *dict, const char *key, double *value) {
    if (!dict || !key || !value) return -1;

    for (size_t i = 0; i < dict->size; i++) {
        if (strcmp(dict->entries[i].key, key) == 0) {
            *value = dict->entries[i].value;
            return 0;
        }
    }

    return -1;  // Key not found
}

int str_dict_contains(StrDict *dict, const char *key) {
    if (!dict || !key) return 0;

    for (size_t i = 0; i < dict->size; i++) {
        if (strcmp(dict->entries[i].key, key) == 0) {
            return 1;
        }
    }

    return 0;
}

void str_dict_print(StrDict *dict) {
    if (!dict) return;

    printf("{\n");
    for (size_t i = 0; i < dict->size; i++) {
        printf("  \"%s\": %f", dict->entries[i].key, dict->entries[i].value);
        if (i < dict->size - 1) printf(",");
        printf("\n");
    }
    printf("}\n");
}

// Python integration function
StrDict* str_dict_from_pydict(PyObject *py_dict) {
    if (!PyDict_Check(py_dict)) {
        PyErr_SetString(PyExc_TypeError, "Expected a dictionary");
        return NULL;
    }

    Py_ssize_t size = PyDict_Size(py_dict);
    StrDict *dict = str_dict_create(size > 0 ? size : 16);
    if (!dict) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate dictionary");
        return NULL;
    }

    PyObject *key, *value;
    Py_ssize_t pos = 0;

    while (PyDict_Next(py_dict, &pos, &key, &value)) {
        // Convert key to string
        const char *key_str = NULL;
        if (PyUnicode_Check(key)) {
            key_str = PyUnicode_AsUTF8(key);
        } else {
            PyErr_SetString(PyExc_TypeError, "All keys must be strings");
            str_dict_free(dict);
            return NULL;
        }

        // Convert value to double
        double val = 0.0;
        if (PyFloat_Check(value)) {
            val = PyFloat_AsDouble(value);
        } else if (PyLong_Check(value)) {
            val = (double)PyLong_AsLong(value);
        } else {
            PyErr_SetString(PyExc_TypeError, "All values must be numeric");
            str_dict_free(dict);
            return NULL;
        }

        // Add to dictionary
        if (str_dict_set(dict, key_str, val) != 0) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to add entry to dictionary");
            str_dict_free(dict);
            return NULL;
        }
    }

    return dict;
}
