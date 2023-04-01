#include <pythonic/core.hpp>
#include <pythonic/python/core.hpp>
#include <pythonic/types/bool.hpp>
#include <pythonic/types/int.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pythonic/include/types/ndarray.hpp>
#include <pythonic/include/types/float64.hpp>
#include <pythonic/types/float64.hpp>
#include <pythonic/types/ndarray.hpp>
#include <pythonic/include/builtins/ValueError.hpp>
#include <pythonic/include/builtins/ZeroDivisionError.hpp>
#include <pythonic/include/builtins/getattr.hpp>
#include <pythonic/include/builtins/range.hpp>
#include <pythonic/include/numpy/arctan2.hpp>
#include <pythonic/include/numpy/cos.hpp>
#include <pythonic/include/numpy/empty_like.hpp>
#include <pythonic/include/numpy/sin.hpp>
#include <pythonic/include/numpy/square.hpp>
#include <pythonic/include/operator_/add.hpp>
#include <pythonic/include/operator_/div.hpp>
#include <pythonic/include/operator_/eq.hpp>
#include <pythonic/include/operator_/iadd.hpp>
#include <pythonic/include/operator_/mul.hpp>
#include <pythonic/include/operator_/ne.hpp>
#include <pythonic/include/operator_/sub.hpp>
#include <pythonic/include/types/slice.hpp>
#include <pythonic/include/types/str.hpp>
#include <pythonic/builtins/ValueError.hpp>
#include <pythonic/builtins/ZeroDivisionError.hpp>
#include <pythonic/builtins/getattr.hpp>
#include <pythonic/builtins/range.hpp>
#include <pythonic/numpy/arctan2.hpp>
#include <pythonic/numpy/cos.hpp>
#include <pythonic/numpy/empty_like.hpp>
#include <pythonic/numpy/sin.hpp>
#include <pythonic/numpy/square.hpp>
#include <pythonic/operator_/add.hpp>
#include <pythonic/operator_/div.hpp>
#include <pythonic/operator_/eq.hpp>
#include <pythonic/operator_/iadd.hpp>
#include <pythonic/operator_/mul.hpp>
#include <pythonic/operator_/ne.hpp>
#include <pythonic/operator_/sub.hpp>
#include <pythonic/types/slice.hpp>
#include <pythonic/types/str.hpp>
namespace __pythran__spectral
{
  struct _lombscargle
  {
    typedef void callable;
    typedef void pure;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 >
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::empty_like{})>::type>::type __type0;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type1;
      typedef typename pythonic::assignable<decltype(std::declval<__type0>()(std::declval<__type1>()))>::type __type2;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::range{})>::type>::type __type3;
      typedef decltype(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, std::declval<__type1>())) __type5;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type5>::type>::type __type6;
      typedef decltype(std::declval<__type3>()(std::declval<__type6>())) __type7;
      typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type7>::type::iterator>::value_type>::type __type8;
      typedef indexable<__type8> __type9;
      typedef typename __combined<__type2,__type9>::type __type10;
      typedef double __type11;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::square{})>::type>::type __type12;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::cos{})>::type>::type __type13;
      typedef decltype(std::declval<__type1>()[std::declval<__type8>()]) __type16;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::arctan2{})>::type>::type __type17;
      typedef long __type18;
      typedef typename pythonic::assignable<double>::type __type19;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type20;
      typedef typename pythonic::assignable<decltype(std::declval<__type0>()(std::declval<__type20>()))>::type __type21;
      typedef decltype(pythonic::operator_::mul(std::declval<__type16>(), std::declval<__type20>())) __type26;
      typedef decltype(std::declval<__type13>()(std::declval<__type26>())) __type27;
      typedef typename __combined<__type21,__type27>::type __type28;
      typedef decltype(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, std::declval<__type20>())) __type30;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type30>::type>::type __type31;
      typedef decltype(std::declval<__type3>()(std::declval<__type31>())) __type32;
      typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type32>::type::iterator>::value_type>::type __type33;
      typedef decltype(std::declval<__type28>()[std::declval<__type33>()]) __type34;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::sin{})>::type>::type __type37;
      typedef decltype(std::declval<__type37>()(std::declval<__type26>())) __type43;
      typedef typename __combined<__type21,__type43>::type __type44;
      typedef decltype(std::declval<__type44>()[std::declval<__type33>()]) __type46;
      typedef decltype(pythonic::operator_::mul(std::declval<__type34>(), std::declval<__type46>())) __type47;
      typedef decltype(pythonic::operator_::add(std::declval<__type19>(), std::declval<__type47>())) __type48;
      typedef typename __combined<__type19,__type48>::type __type49;
      typedef typename __combined<__type49,__type47>::type __type50;
      typedef decltype(pythonic::operator_::mul(std::declval<__type18>(), std::declval<__type50>())) __type51;
      typedef decltype(std::declval<__type12>()(std::declval<__type34>())) __type55;
      typedef decltype(pythonic::operator_::add(std::declval<__type19>(), std::declval<__type55>())) __type56;
      typedef typename __combined<__type19,__type56>::type __type57;
      typedef typename __combined<__type57,__type55>::type __type58;
      typedef decltype(std::declval<__type12>()(std::declval<__type46>())) __type62;
      typedef decltype(pythonic::operator_::add(std::declval<__type19>(), std::declval<__type62>())) __type63;
      typedef typename __combined<__type19,__type63>::type __type64;
      typedef typename __combined<__type64,__type62>::type __type65;
      typedef decltype(pythonic::operator_::sub(std::declval<__type58>(), std::declval<__type65>())) __type66;
      typedef decltype(std::declval<__type17>()(std::declval<__type51>(), std::declval<__type66>())) __type67;
      typedef decltype(pythonic::operator_::mul(std::declval<__type18>(), std::declval<__type16>())) __type71;
      typedef typename pythonic::assignable<decltype(pythonic::operator_::div(std::declval<__type67>(), std::declval<__type71>()))>::type __type72;
      typedef decltype(pythonic::operator_::mul(std::declval<__type16>(), std::declval<__type72>())) __type73;
      typedef typename pythonic::assignable<decltype(std::declval<__type13>()(std::declval<__type73>()))>::type __type74;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type75;
      typedef decltype(std::declval<__type75>()[std::declval<__type33>()]) __type77;
      typedef decltype(pythonic::operator_::mul(std::declval<__type77>(), std::declval<__type34>())) __type81;
      typedef decltype(pythonic::operator_::add(std::declval<__type19>(), std::declval<__type81>())) __type82;
      typedef typename __combined<__type19,__type82>::type __type83;
      typedef typename __combined<__type83,__type81>::type __type84;
      typedef decltype(pythonic::operator_::mul(std::declval<__type74>(), std::declval<__type84>())) __type85;
      typedef typename pythonic::assignable<decltype(std::declval<__type37>()(std::declval<__type73>()))>::type __type91;
      typedef decltype(pythonic::operator_::mul(std::declval<__type77>(), std::declval<__type46>())) __type98;
      typedef decltype(pythonic::operator_::add(std::declval<__type19>(), std::declval<__type98>())) __type99;
      typedef typename __combined<__type19,__type99>::type __type100;
      typedef typename __combined<__type100,__type98>::type __type101;
      typedef decltype(pythonic::operator_::mul(std::declval<__type91>(), std::declval<__type101>())) __type102;
      typedef decltype(pythonic::operator_::add(std::declval<__type85>(), std::declval<__type102>())) __type103;
      typedef decltype(std::declval<__type12>()(std::declval<__type103>())) __type104;
      typedef typename pythonic::assignable<decltype(std::declval<__type12>()(std::declval<__type74>()))>::type __type106;
      typedef decltype(pythonic::operator_::mul(std::declval<__type106>(), std::declval<__type58>())) __type108;
      typedef decltype(pythonic::operator_::mul(std::declval<__type18>(), std::declval<__type74>())) __type110;
      typedef typename pythonic::assignable<decltype(pythonic::operator_::mul(std::declval<__type110>(), std::declval<__type91>()))>::type __type112;
      typedef decltype(pythonic::operator_::mul(std::declval<__type112>(), std::declval<__type50>())) __type114;
      typedef decltype(pythonic::operator_::add(std::declval<__type108>(), std::declval<__type114>())) __type115;
      typedef typename pythonic::assignable<decltype(std::declval<__type12>()(std::declval<__type91>()))>::type __type117;
      typedef decltype(pythonic::operator_::mul(std::declval<__type117>(), std::declval<__type65>())) __type119;
      typedef decltype(pythonic::operator_::add(std::declval<__type115>(), std::declval<__type119>())) __type120;
      typedef decltype(pythonic::operator_::div(std::declval<__type104>(), std::declval<__type120>())) __type121;
      typedef decltype(pythonic::operator_::mul(std::declval<__type74>(), std::declval<__type101>())) __type124;
      typedef decltype(pythonic::operator_::mul(std::declval<__type91>(), std::declval<__type84>())) __type127;
      typedef decltype(pythonic::operator_::sub(std::declval<__type124>(), std::declval<__type127>())) __type128;
      typedef decltype(std::declval<__type12>()(std::declval<__type128>())) __type129;
      typedef decltype(pythonic::operator_::mul(std::declval<__type106>(), std::declval<__type65>())) __type132;
      typedef decltype(pythonic::operator_::sub(std::declval<__type132>(), std::declval<__type114>())) __type136;
      typedef decltype(pythonic::operator_::mul(std::declval<__type117>(), std::declval<__type58>())) __type139;
      typedef decltype(pythonic::operator_::add(std::declval<__type136>(), std::declval<__type139>())) __type140;
      typedef decltype(pythonic::operator_::div(std::declval<__type129>(), std::declval<__type140>())) __type141;
      typedef decltype(pythonic::operator_::add(std::declval<__type121>(), std::declval<__type141>())) __type142;
      typedef decltype(pythonic::operator_::mul(std::declval<__type11>(), std::declval<__type142>())) __type143;
      typedef container<typename std::remove_reference<__type143>::type> __type144;
      typedef typename pythonic::returnable<typename __combined<__type10,__type144,__type9>::type>::type result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 >
    typename type<argument_type0, argument_type1, argument_type2>::result_type operator()(argument_type0&& x, argument_type1&& y, argument_type2&& freqs) const
    ;
  }  ;
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 >
  typename _lombscargle::type<argument_type0, argument_type1, argument_type2>::result_type _lombscargle::operator()(argument_type0&& x, argument_type1&& y, argument_type2&& freqs) const
  {
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::empty_like{})>::type>::type __type0;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type1;
    typedef typename pythonic::assignable<decltype(std::declval<__type0>()(std::declval<__type1>()))>::type __type2;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::range{})>::type>::type __type3;
    typedef decltype(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, std::declval<__type1>())) __type5;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type5>::type>::type __type6;
    typedef decltype(std::declval<__type3>()(std::declval<__type6>())) __type7;
    typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type7>::type::iterator>::value_type>::type __type8;
    typedef indexable<__type8> __type9;
    typedef typename __combined<__type2,__type9>::type __type10;
    typedef double __type11;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::square{})>::type>::type __type12;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::cos{})>::type>::type __type13;
    typedef decltype(std::declval<__type1>()[std::declval<__type8>()]) __type16;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::arctan2{})>::type>::type __type17;
    typedef long __type18;
    typedef typename pythonic::assignable<double>::type __type19;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type20;
    typedef typename pythonic::assignable<decltype(std::declval<__type0>()(std::declval<__type20>()))>::type __type21;
    typedef decltype(pythonic::operator_::mul(std::declval<__type16>(), std::declval<__type20>())) __type26;
    typedef decltype(std::declval<__type13>()(std::declval<__type26>())) __type27;
    typedef typename __combined<__type21,__type27>::type __type28;
    typedef decltype(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, std::declval<__type20>())) __type30;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type30>::type>::type __type31;
    typedef decltype(std::declval<__type3>()(std::declval<__type31>())) __type32;
    typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type32>::type::iterator>::value_type>::type __type33;
    typedef decltype(std::declval<__type28>()[std::declval<__type33>()]) __type34;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::sin{})>::type>::type __type37;
    typedef decltype(std::declval<__type37>()(std::declval<__type26>())) __type43;
    typedef typename __combined<__type21,__type43>::type __type44;
    typedef decltype(std::declval<__type44>()[std::declval<__type33>()]) __type46;
    typedef decltype(pythonic::operator_::mul(std::declval<__type34>(), std::declval<__type46>())) __type47;
    typedef decltype(pythonic::operator_::add(std::declval<__type19>(), std::declval<__type47>())) __type48;
    typedef typename __combined<__type19,__type48>::type __type49;
    typedef typename __combined<__type49,__type47>::type __type50;
    typedef decltype(pythonic::operator_::mul(std::declval<__type18>(), std::declval<__type50>())) __type51;
    typedef decltype(std::declval<__type12>()(std::declval<__type34>())) __type55;
    typedef decltype(pythonic::operator_::add(std::declval<__type19>(), std::declval<__type55>())) __type56;
    typedef typename __combined<__type19,__type56>::type __type57;
    typedef typename __combined<__type57,__type55>::type __type58;
    typedef decltype(std::declval<__type12>()(std::declval<__type46>())) __type62;
    typedef decltype(pythonic::operator_::add(std::declval<__type19>(), std::declval<__type62>())) __type63;
    typedef typename __combined<__type19,__type63>::type __type64;
    typedef typename __combined<__type64,__type62>::type __type65;
    typedef decltype(pythonic::operator_::sub(std::declval<__type58>(), std::declval<__type65>())) __type66;
    typedef decltype(std::declval<__type17>()(std::declval<__type51>(), std::declval<__type66>())) __type67;
    typedef decltype(pythonic::operator_::mul(std::declval<__type18>(), std::declval<__type16>())) __type71;
    typedef typename pythonic::assignable<decltype(pythonic::operator_::div(std::declval<__type67>(), std::declval<__type71>()))>::type __type72;
    typedef decltype(pythonic::operator_::mul(std::declval<__type16>(), std::declval<__type72>())) __type73;
    typedef typename pythonic::assignable<decltype(std::declval<__type13>()(std::declval<__type73>()))>::type __type74;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type75;
    typedef decltype(std::declval<__type75>()[std::declval<__type33>()]) __type77;
    typedef decltype(pythonic::operator_::mul(std::declval<__type77>(), std::declval<__type34>())) __type81;
    typedef decltype(pythonic::operator_::add(std::declval<__type19>(), std::declval<__type81>())) __type82;
    typedef typename __combined<__type19,__type82>::type __type83;
    typedef typename __combined<__type83,__type81>::type __type84;
    typedef decltype(pythonic::operator_::mul(std::declval<__type74>(), std::declval<__type84>())) __type85;
    typedef typename pythonic::assignable<decltype(std::declval<__type37>()(std::declval<__type73>()))>::type __type91;
    typedef decltype(pythonic::operator_::mul(std::declval<__type77>(), std::declval<__type46>())) __type98;
    typedef decltype(pythonic::operator_::add(std::declval<__type19>(), std::declval<__type98>())) __type99;
    typedef typename __combined<__type19,__type99>::type __type100;
    typedef typename __combined<__type100,__type98>::type __type101;
    typedef decltype(pythonic::operator_::mul(std::declval<__type91>(), std::declval<__type101>())) __type102;
    typedef decltype(pythonic::operator_::add(std::declval<__type85>(), std::declval<__type102>())) __type103;
    typedef decltype(std::declval<__type12>()(std::declval<__type103>())) __type104;
    typedef typename pythonic::assignable<decltype(std::declval<__type12>()(std::declval<__type74>()))>::type __type106;
    typedef decltype(pythonic::operator_::mul(std::declval<__type106>(), std::declval<__type58>())) __type108;
    typedef decltype(pythonic::operator_::mul(std::declval<__type18>(), std::declval<__type74>())) __type110;
    typedef typename pythonic::assignable<decltype(pythonic::operator_::mul(std::declval<__type110>(), std::declval<__type91>()))>::type __type112;
    typedef decltype(pythonic::operator_::mul(std::declval<__type112>(), std::declval<__type50>())) __type114;
    typedef decltype(pythonic::operator_::add(std::declval<__type108>(), std::declval<__type114>())) __type115;
    typedef typename pythonic::assignable<decltype(std::declval<__type12>()(std::declval<__type91>()))>::type __type117;
    typedef decltype(pythonic::operator_::mul(std::declval<__type117>(), std::declval<__type65>())) __type119;
    typedef decltype(pythonic::operator_::add(std::declval<__type115>(), std::declval<__type119>())) __type120;
    typedef decltype(pythonic::operator_::div(std::declval<__type104>(), std::declval<__type120>())) __type121;
    typedef decltype(pythonic::operator_::mul(std::declval<__type74>(), std::declval<__type101>())) __type124;
    typedef decltype(pythonic::operator_::mul(std::declval<__type91>(), std::declval<__type84>())) __type127;
    typedef decltype(pythonic::operator_::sub(std::declval<__type124>(), std::declval<__type127>())) __type128;
    typedef decltype(std::declval<__type12>()(std::declval<__type128>())) __type129;
    typedef decltype(pythonic::operator_::mul(std::declval<__type106>(), std::declval<__type65>())) __type132;
    typedef decltype(pythonic::operator_::sub(std::declval<__type132>(), std::declval<__type114>())) __type136;
    typedef decltype(pythonic::operator_::mul(std::declval<__type117>(), std::declval<__type58>())) __type139;
    typedef decltype(pythonic::operator_::add(std::declval<__type136>(), std::declval<__type139>())) __type140;
    typedef decltype(pythonic::operator_::div(std::declval<__type129>(), std::declval<__type140>())) __type141;
    typedef decltype(pythonic::operator_::add(std::declval<__type121>(), std::declval<__type141>())) __type142;
    typedef decltype(pythonic::operator_::mul(std::declval<__type11>(), std::declval<__type142>())) __type143;
    typedef container<typename std::remove_reference<__type143>::type> __type144;
    typename pythonic::assignable<typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type32>::type::iterator>::value_type>::type>::type j;
    typename pythonic::assignable<typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type7>::type::iterator>::value_type>::type>::type i;
    if (pythonic::operator_::ne(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, x), pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, y)))
    {
      throw pythonic::builtins::functor::ValueError{}(pythonic::types::str("Input arrays do not have the same size."));
    }
    typename pythonic::assignable<typename __combined<__type10,__type144,__type9>::type>::type pgram = pythonic::numpy::functor::empty_like{}(freqs);
    typename pythonic::assignable<typename __combined<__type21,__type27>::type>::type c = pythonic::numpy::functor::empty_like{}(x);
    typename pythonic::assignable<typename __combined<__type21,__type43>::type>::type s = pythonic::numpy::functor::empty_like{}(x);
    {
      long  __target139838532253776 = std::get<0>(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, freqs));
      for (long  i=0L; i < __target139838532253776; i += 1L)
      {
        typename pythonic::assignable<typename __combined<__type83,__type81>::type>::type xc = 0.0;
        typename pythonic::assignable<typename __combined<__type100,__type98>::type>::type xs = 0.0;
        typename pythonic::assignable<typename __combined<__type57,__type55>::type>::type cc = 0.0;
        typename pythonic::assignable<typename __combined<__type64,__type62>::type>::type ss = 0.0;
        typename pythonic::assignable<typename __combined<__type49,__type47>::type>::type cs = 0.0;
        c[pythonic::types::contiguous_slice(pythonic::builtins::None,pythonic::builtins::None)] = pythonic::numpy::functor::cos{}(pythonic::operator_::mul(freqs.fast(i), x));
        s[pythonic::types::contiguous_slice(pythonic::builtins::None,pythonic::builtins::None)] = pythonic::numpy::functor::sin{}(pythonic::operator_::mul(freqs.fast(i), x));
        {
          long  __target139838528169200 = std::get<0>(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, x));
          for (long  j=0L; j < __target139838528169200; j += 1L)
          {
            xc += pythonic::operator_::mul(y.fast(j), c.fast(j));
            xs += pythonic::operator_::mul(y.fast(j), s.fast(j));
            cc += pythonic::numpy::functor::square{}(c.fast(j));
            ss += pythonic::numpy::functor::square{}(s.fast(j));
            cs += pythonic::operator_::mul(c.fast(j), s.fast(j));
          }
        }
        if (pythonic::operator_::eq(freqs.fast(i), 0L))
        {
          throw pythonic::builtins::functor::ZeroDivisionError{}();
        }
        typename pythonic::assignable_noescape<decltype(pythonic::operator_::div(pythonic::numpy::functor::arctan2{}(pythonic::operator_::mul(2L, cs), pythonic::operator_::sub(cc, ss)), pythonic::operator_::mul(2L, freqs.fast(i))))>::type tau = pythonic::operator_::div(pythonic::numpy::functor::arctan2{}(pythonic::operator_::mul(2L, cs), pythonic::operator_::sub(cc, ss)), pythonic::operator_::mul(2L, freqs.fast(i)));
        typename pythonic::assignable_noescape<decltype(pythonic::numpy::functor::cos{}(pythonic::operator_::mul(freqs.fast(i), tau)))>::type c_tau = pythonic::numpy::functor::cos{}(pythonic::operator_::mul(freqs.fast(i), tau));
        typename pythonic::assignable_noescape<decltype(pythonic::numpy::functor::sin{}(pythonic::operator_::mul(freqs.fast(i), tau)))>::type s_tau = pythonic::numpy::functor::sin{}(pythonic::operator_::mul(freqs.fast(i), tau));
        typename pythonic::assignable_noescape<decltype(pythonic::numpy::functor::square{}(c_tau))>::type c_tau2 = pythonic::numpy::functor::square{}(c_tau);
        typename pythonic::assignable_noescape<decltype(pythonic::numpy::functor::square{}(s_tau))>::type s_tau2 = pythonic::numpy::functor::square{}(s_tau);
        typename pythonic::assignable_noescape<decltype(pythonic::operator_::mul(pythonic::operator_::mul(2L, c_tau), s_tau))>::type cs_tau = pythonic::operator_::mul(pythonic::operator_::mul(2L, c_tau), s_tau);
        pgram.fast(i) = pythonic::operator_::mul(0.5, pythonic::operator_::add(pythonic::operator_::div(pythonic::numpy::functor::square{}(pythonic::operator_::add(pythonic::operator_::mul(c_tau, xc), pythonic::operator_::mul(s_tau, xs))), pythonic::operator_::add(pythonic::operator_::add(pythonic::operator_::mul(c_tau2, cc), pythonic::operator_::mul(cs_tau, cs)), pythonic::operator_::mul(s_tau2, ss))), pythonic::operator_::div(pythonic::numpy::functor::square{}(pythonic::operator_::sub(pythonic::operator_::mul(c_tau, xs), pythonic::operator_::mul(s_tau, xc))), pythonic::operator_::add(pythonic::operator_::sub(pythonic::operator_::mul(c_tau2, ss), pythonic::operator_::mul(cs_tau, cs)), pythonic::operator_::mul(s_tau2, cc)))));
      }
    }
    return pgram;
  }
}
#include <pythonic/python/exception_handler.hpp>
#ifdef ENABLE_PYTHON_MODULE
typename __pythran__spectral::_lombscargle::type<pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>>::result_type _lombscargle0(pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& x, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& y, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& freqs) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__spectral::_lombscargle()(x, y, freqs);
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
__pythran_wrap__lombscargle0(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[3+1];
    char const* keywords[] = {"x", "y", "freqs",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2]))
        return to_python(_lombscargle0(from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[0]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[1]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2])));
    else {
        return nullptr;
    }
}

            static PyObject *
            __pythran_wrapall__lombscargle(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap__lombscargle0(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "_lombscargle", "\n""    - _lombscargle(float64[:], float64[:], float64[:])", args, kw);
                });
            }


static PyMethodDef Methods[] = {
    {
    "_lombscargle",
    (PyCFunction)__pythran_wrapall__lombscargle,
    METH_VARARGS | METH_KEYWORDS,
    "\n""_lombscargle(x, y, freqs)\n""\n""Supported prototypes:\n""\n""- _lombscargle(float64[:], float64[:], float64[:])\n""\n""Computes the Lomb-Scargle periodogram.\n""\n""Parameters\n""----------\n""x : array_like\n""    Sample times.\n""y : array_like\n""    Measurement values (must be registered so the mean is zero).\n""freqs : array_like\n""    Angular frequencies for output periodogram.\n""\n""Returns\n""-------\n""pgram : array_like\n""    Lomb-Scargle periodogram.\n""\n""Raises\n""------\n""ValueError\n""    If the input arrays `x` and `y` do not have the same shape.\n""\n""See also\n""--------\n""lombscargle\n""\n"""},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_spectral",            /* m_name */
    "Tools for spectral analysis of unequally sampled signals.",         /* m_doc */
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
PYTHRAN_MODULE_INIT(_spectral)(void)
#ifndef _WIN32
__attribute__ ((visibility("default")))
__attribute__ ((externally_visible))
#endif
;
PyMODINIT_FUNC
PYTHRAN_MODULE_INIT(_spectral)(void) {
    import_array()
    #if PY_MAJOR_VERSION >= 3
    PyObject* theModule = PyModule_Create(&moduledef);
    #else
    PyObject* theModule = Py_InitModule3("_spectral",
                                         Methods,
                                         "Tools for spectral analysis of unequally sampled signals."
    );
    #endif
    if(! theModule)
        PYTHRAN_RETURN;
    PyObject * theDoc = Py_BuildValue("(sss)",
                                      "0.9.11",
                                      "2021-08-01 14:59:06.196980",
                                      "a6baed4d2848dab8f54758df4dab62a2bafc41cd4f7be2b4df2c574a78e74b15");
    if(! theDoc)
        PYTHRAN_RETURN;
    PyModule_AddObject(theModule,
                       "__pythran__",
                       theDoc);


    PYTHRAN_RETURN;
}

#endif