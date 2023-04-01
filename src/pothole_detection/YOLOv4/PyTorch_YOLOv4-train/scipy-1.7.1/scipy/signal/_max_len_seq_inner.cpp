#include <pythonic/core.hpp>
#include <pythonic/python/core.hpp>
#include <pythonic/types/bool.hpp>
#include <pythonic/types/int.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pythonic/include/types/int8.hpp>
#include <pythonic/include/types/ndarray.hpp>
#include <pythonic/include/types/int.hpp>
#include <pythonic/include/types/int64.hpp>
#include <pythonic/types/int64.hpp>
#include <pythonic/types/int8.hpp>
#include <pythonic/types/ndarray.hpp>
#include <pythonic/types/int.hpp>
#include <pythonic/include/builtins/getattr.hpp>
#include <pythonic/include/builtins/range.hpp>
#include <pythonic/include/numpy/roll.hpp>
#include <pythonic/include/operator_/add.hpp>
#include <pythonic/include/operator_/ixor.hpp>
#include <pythonic/include/operator_/mod.hpp>
#include <pythonic/include/operator_/neg.hpp>
#include <pythonic/include/operator_/xor_.hpp>
#include <pythonic/include/types/str.hpp>
#include <pythonic/builtins/getattr.hpp>
#include <pythonic/builtins/range.hpp>
#include <pythonic/numpy/roll.hpp>
#include <pythonic/operator_/add.hpp>
#include <pythonic/operator_/ixor.hpp>
#include <pythonic/operator_/mod.hpp>
#include <pythonic/operator_/neg.hpp>
#include <pythonic/operator_/xor_.hpp>
#include <pythonic/types/str.hpp>
namespace __pythran__max_len_seq_inner
{
  struct _max_len_seq_inner
  {
    typedef void callable;
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 >
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type0;
      typedef typename pythonic::assignable<long>::type __type1;
      typedef decltype(std::declval<__type0>()[std::declval<__type1>()]) __type2;
      typedef typename pythonic::assignable<decltype(std::declval<__type0>()[std::declval<__type1>()])>::type __type3;
      typedef container<typename std::remove_reference<__type2>::type> __type5;
      typedef typename __combined<__type0,__type5>::type __type6;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type7;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::range{})>::type>::type __type8;
      typedef decltype(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, std::declval<__type7>())) __type10;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type10>::type>::type __type11;
      typedef typename pythonic::lazy<__type11>::type __type12;
      typedef decltype(std::declval<__type8>()(std::declval<__type12>())) __type13;
      typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type13>::type::iterator>::value_type>::type __type14;
      typedef decltype(std::declval<__type7>()[std::declval<__type14>()]) __type15;
      typedef decltype(pythonic::operator_::add(std::declval<__type15>(), std::declval<__type1>())) __type17;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type18;
      typedef decltype(pythonic::operator_::mod(std::declval<__type17>(), std::declval<__type18>())) __type19;
      typedef decltype(std::declval<__type6>()[std::declval<__type19>()]) __type20;
      typedef typename __combined<__type3,__type20>::type __type21;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::roll{})>::type>::type __type22;
      typedef container<typename std::remove_reference<__type20>::type> __type23;
      typedef container<typename std::remove_reference<__type21>::type> __type25;
      typedef indexable<__type1> __type27;
      typedef typename __combined<__type0,__type5,__type23,__type25,__type27>::type __type28;
      typedef long __type31;
      typedef decltype(pythonic::operator_::add(std::declval<__type1>(), std::declval<__type31>())) __type32;
      typedef typename pythonic::assignable<decltype(pythonic::operator_::mod(std::declval<__type32>(), std::declval<__type18>()))>::type __type34;
      typedef typename __combined<__type1,__type34>::type __type35;
      typedef decltype(pythonic::operator_::neg(std::declval<__type35>())) __type36;
      typedef __type2 __ptype0;
      typedef __type21 __ptype1;
      typedef typename pythonic::returnable<decltype(std::declval<__type22>()(std::declval<__type28>(), std::declval<__type36>(), std::declval<__type31>()))>::type result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 >
    typename type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4>::result_type operator()(argument_type0&& taps, argument_type1&& state, argument_type2&& nbits, argument_type3&& length, argument_type4&& seq) const
    ;
  }  ;
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 >
  typename _max_len_seq_inner::type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4>::result_type _max_len_seq_inner::operator()(argument_type0&& taps, argument_type1&& state, argument_type2&& nbits, argument_type3&& length, argument_type4&& seq) const
  {
    typedef typename pythonic::assignable<long>::type __type0;
    typedef long __type2;
    typedef decltype(pythonic::operator_::add(std::declval<__type0>(), std::declval<__type2>())) __type3;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type4;
    typedef typename pythonic::assignable<decltype(pythonic::operator_::mod(std::declval<__type3>(), std::declval<__type4>()))>::type __type5;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type6;
    typedef typename __combined<__type0,__type5>::type __type8;
    typedef decltype(std::declval<__type6>()[std::declval<__type8>()]) __type9;
    typedef container<typename std::remove_reference<__type9>::type> __type10;
    typedef typename __combined<__type6,__type10>::type __type11;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type12;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::range{})>::type>::type __type13;
    typedef decltype(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, std::declval<__type12>())) __type15;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type15>::type>::type __type16;
    typedef typename pythonic::lazy<__type16>::type __type17;
    typedef decltype(std::declval<__type13>()(std::declval<__type17>())) __type18;
    typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type18>::type::iterator>::value_type>::type __type19;
    typedef decltype(std::declval<__type12>()[std::declval<__type19>()]) __type20;
    typedef decltype(pythonic::operator_::add(std::declval<__type20>(), std::declval<__type8>())) __type22;
    typedef decltype(pythonic::operator_::mod(std::declval<__type22>(), std::declval<__type4>())) __type24;
    typedef decltype(std::declval<__type11>()[std::declval<__type24>()]) __type25;
    typedef container<typename std::remove_reference<__type25>::type> __type26;
    typedef typename pythonic::assignable<decltype(std::declval<__type6>()[std::declval<__type8>()])>::type __type27;
    typedef typename __combined<__type27,__type25>::type __type28;
    typedef container<typename std::remove_reference<__type28>::type> __type29;
    typedef indexable<__type8> __type31;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type3>::type>::type __type32;
    typedef decltype(std::declval<__type13>()(std::declval<__type32>())) __type33;
    typename pythonic::assignable<typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type18>::type::iterator>::value_type>::type>::type ti;
    typename pythonic::assignable<typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type33>::type::iterator>::value_type>::type>::type i;
    typename pythonic::lazy<decltype(std::get<0>(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, taps)))>::type n_taps = std::get<0>(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, taps));
    typename pythonic::assignable<typename __combined<__type0,__type5>::type>::type idx = 0L;
    {
      long  __target139838532972448 = length;
      for (long  i=0L; i < __target139838532972448; i += 1L)
      {
        typename pythonic::assignable<typename __combined<__type27,__type25>::type>::type feedback = state.fast(idx);
        seq.fast(i) = feedback;
        {
          long  __target139838532969952 = n_taps;
          for (long  ti=0L; ti < __target139838532969952; ti += 1L)
          {
            feedback ^= state.fast(pythonic::operator_::mod(pythonic::operator_::add(taps.fast(ti), idx), nbits));
          }
        }
        state.fast(idx) = feedback;
        idx = pythonic::operator_::mod(pythonic::operator_::add(idx, 1L), nbits);
      }
    }
    return pythonic::numpy::functor::roll{}(state, pythonic::operator_::neg(idx), 0L);
  }
}
#include <pythonic/python/exception_handler.hpp>
#ifdef ENABLE_PYTHON_MODULE
typename __pythran__max_len_seq_inner::_max_len_seq_inner::type<pythonic::types::ndarray<npy_int64,pythonic::types::pshape<long>>, pythonic::types::ndarray<npy_int8,pythonic::types::pshape<long>>, long, long, pythonic::types::ndarray<npy_int8,pythonic::types::pshape<long>>>::result_type _max_len_seq_inner0(pythonic::types::ndarray<npy_int64,pythonic::types::pshape<long>>&& taps, pythonic::types::ndarray<npy_int8,pythonic::types::pshape<long>>&& state, long&& nbits, long&& length, pythonic::types::ndarray<npy_int8,pythonic::types::pshape<long>>&& seq) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__max_len_seq_inner::_max_len_seq_inner()(taps, state, nbits, length, seq);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}

static PyObject *
__pythran_wrap__max_len_seq_inner0(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[5+1];
    char const* keywords[] = {"taps", "state", "nbits", "length", "seq",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<npy_int64,pythonic::types::pshape<long>>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<npy_int8,pythonic::types::pshape<long>>>(args_obj[1]) && is_convertible<long>(args_obj[2]) && is_convertible<long>(args_obj[3]) && is_convertible<pythonic::types::ndarray<npy_int8,pythonic::types::pshape<long>>>(args_obj[4]))
        return to_python(_max_len_seq_inner0(from_python<pythonic::types::ndarray<npy_int64,pythonic::types::pshape<long>>>(args_obj[0]), from_python<pythonic::types::ndarray<npy_int8,pythonic::types::pshape<long>>>(args_obj[1]), from_python<long>(args_obj[2]), from_python<long>(args_obj[3]), from_python<pythonic::types::ndarray<npy_int8,pythonic::types::pshape<long>>>(args_obj[4])));
    else {
        return nullptr;
    }
}

            static PyObject *
            __pythran_wrapall__max_len_seq_inner(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap__max_len_seq_inner0(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "_max_len_seq_inner", "\n""    - _max_len_seq_inner(int64[:], int8[:], int, int, int8[:])", args, kw);
                });
            }


static PyMethodDef Methods[] = {
    {
    "_max_len_seq_inner",
    (PyCFunction)__pythran_wrapall__max_len_seq_inner,
    METH_VARARGS | METH_KEYWORDS,
    "Supported prototypes:\n""\n""    - _max_len_seq_inner(int64[:], int8[:], int, int, int8[:])"},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_max_len_seq_inner",            /* m_name */
    "",         /* m_doc */
    -1,                  /* m_size */
    Methods,             /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
  };
#define PYTHRAN_RETURN return theModule
#define PYTHRAN_MODULE_INIT(s) PyInit_##s
#else
#define PYTHRAN_RETURN return
#define PYTHRAN_MODULE_INIT(s) init##s
#endif
PyMODINIT_FUNC
PYTHRAN_MODULE_INIT(_max_len_seq_inner)(void)
#ifndef _WIN32
__attribute__ ((visibility("default")))
__attribute__ ((externally_visible))
#endif
;
PyMODINIT_FUNC
PYTHRAN_MODULE_INIT(_max_len_seq_inner)(void) {
    import_array()
    #if PY_MAJOR_VERSION >= 3
    PyObject* theModule = PyModule_Create(&moduledef);
    #else
    PyObject* theModule = Py_InitModule3("_max_len_seq_inner",
                                         Methods,
                                         ""
    );
    #endif
    if(! theModule)
        PYTHRAN_RETURN;
    PyObject * theDoc = Py_BuildValue("(sss)",
                                      "0.9.11",
                                      "2021-08-01 14:59:05.891799",
                                      "ccd4c6b01dd37c8a698a8beaea85985732ea19d0ab7d0883d689f577fbda0424");
    if(! theDoc)
        PYTHRAN_RETURN;
    PyModule_AddObject(theModule,
                       "__pythran__",
                       theDoc);


    PYTHRAN_RETURN;
}

#endif