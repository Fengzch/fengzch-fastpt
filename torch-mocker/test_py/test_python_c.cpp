// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

// #include <pybind11/pybind11.h>
// #include <torch/csrc/THP.h>
// #include <torch/csrc/Stream.h>
// #include <torch/csrc/cuda/Stream.h>
// #include <torch/csrc/utils.h>

// /*******注册THPUtils_PySequence_to_CUDAStreamList**********/
// static PyObject* py_THPUtils_PySequence_to_CUDAStreamList(PyObject* self, PyObject* args) {
//     PyObject* obj;
//     if (!PyArg_ParseTuple(args, "O", &obj)) {
//         return nullptr;
//     }
//     try {
//         auto streams = THPUtils_PySequence_to_CUDAStreamList(obj);
//         PyObject* py_list = PyList_New(streams.size());
//         if (!py_list) {
//             return nullptr;
//         }
//         for (size_t i = 0; i < streams.size(); ++i) {
//             PyObject* py_stream = nullptr;
//             if (streams[i].has_value()) {
//                 py_stream = PyLong_FromVoidPtr(reinterpret_cast<void*>(streams[i]->stream()));
//             } else {
//                 Py_INCREF(Py_None);
//                 py_stream = Py_None;
//             }
//             PyList_SET_ITEM(py_list, i, py_stream);
//         }
//         return py_list;
//     } catch (const std::runtime_error& e) {
//         PyErr_SetString(PyExc_RuntimeError, e.what());
//         return nullptr;
//     }
// }
// static PyMethodDef UtilMethods[] = {
//     {"to_cuda_stream_list", (PyCFunction)py_THPUtils_PySequence_to_CUDAStreamList, METH_VARARGS, "Convert a Python sequence to CUDAStream list"},
//     {nullptr, nullptr, 0, nullptr}
// };

// void utils_init(PyObject* module) {
//     for (PyMethodDef* method = UtilMethods; method->ml_name != nullptr; ++method) {
//         PyObject* func = PyCFunction_NewEx(method, nullptr, nullptr);
//         if (func == nullptr || PyModule_AddObject(module, method->ml_name, func) < 0) {
//             Py_XDECREF(func);
//             throw python_error();
//         }
//     }
// }

// /*******注册THPUtils_PySequence_to_CUDAStreamList**********/

// static struct PyModuleDef stream_module = {
//     PyModuleDef_HEAD_INIT,
//     "test_THCP",
//     "Test THCP Module",
//     -1,
//     nullptr
// };

// PyMODINIT_FUNC PyInit_test_THCP(void) {
//     PyObject* module = PyModule_Create(&stream_module);
//     if (!module) {
//         return nullptr;
//     }
//     THPStream_init(module);//先初始化THPStreamClass
//     THCPStream_init(module);
//     utils_init(module);
//     return module;
// }