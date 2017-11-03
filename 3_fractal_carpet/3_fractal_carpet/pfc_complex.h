//       $Id: pfc_complex.h 1306 2017-11-03 09:43:10Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/se/sw/mpv3/trunk/Lecture/Source/ACC/common/pfc_complex.h $
// $Revision: 1306 $
//     $Date: 2017-11-03 10:43:10 +0100 (Fr., 03 Nov 2017) $
//   Creator: peter.kulczycki<AT>fh-hagenberg.at
//   $Author: p20068 $
//
// Copyright: (c) 2017 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: Distributed under the Boost Software License, Version 1.0 (see
//            http://www.boost.org/LICENSE_1_0.txt).

#if !defined PFC_COMPLEX_H
#define      PFC_COMPLEX_H

#include "./pfc_cuda_macros.h"
#include "./pfc_traits.h"

namespace pfc {

// -------------------------------------------------------------------------------------------------

template <typename T = double> class complex final {
   public:
      using imag_t  = T;
      using real_t  = T;
      using value_t = T;

      static_assert (
         std::is_integral_v <value_t> || std::is_floating_point_v <value_t>, "value_t must be an integral or a floating-point type"
      );

      /*CATTR_HOST_DEV*/ constexpr complex () = default;

      CATTR_HOST_DEV constexpr complex (value_t const r) : real {r} {
      }

      CATTR_HOST_DEV constexpr complex (value_t const r, value_t const i) : real {r}, imag {i} {
      }

      /*CATTR_HOST_DEV*/ complex (complex const &) = default;
      /*CATTR_HOST_DEV*/ complex (complex &&) = default;

      /*CATTR_HOST_DEV*/ complex & operator = (complex const &) = default;
      /*CATTR_HOST_DEV*/ complex & operator = (complex &&) = default;

      CATTR_HOST_DEV complex & operator += (complex const & rhs) {
         real += rhs.real; imag += rhs.imag; return *this;
      }

      CATTR_HOST_DEV complex & operator -= (complex const & rhs) {
         real -= rhs.real; imag -= rhs.imag; return *this;
      }

      CATTR_HOST_DEV complex operator + (complex const & rhs) const {
         auto lhs {*this}; return lhs += rhs;
      }

      CATTR_HOST_DEV complex operator - (complex const & rhs) const {
         auto lhs {*this}; return lhs -= rhs;
      }

      CATTR_HOST_DEV complex operator - () const {
         return complex {} -= *this;
      }

      CATTR_HOST_DEV complex operator * (complex const & rhs) const {
         return {real * rhs.real - imag * rhs.imag, imag * rhs.real + real * rhs.imag};
      }

      CATTR_HOST_DEV value_t norm () const {
         return real * real + imag * imag;
      }

      CATTR_HOST_DEV complex & square () {
         auto const r {real * real - imag * imag};

         imag = real * imag * 2;
         real = r;

         return *this;
      }

      value_t real {};
      value_t imag {};
};

// -------------------------------------------------------------------------------------------------

template <typename value_t> CATTR_HOST_DEV constexpr auto operator + (value_t const lhs, pfc::complex <value_t> const & rhs) {
   return pfc::complex <value_t> {lhs + rhs.real, rhs.imag};
}

template <typename value_t> CATTR_HOST_DEV constexpr auto operator * (value_t const lhs, pfc::complex <value_t> const & rhs) {
   return pfc::complex <value_t> {lhs * rhs.real, lhs * rhs.imag};
}

template <typename value_t> CATTR_HOST_DEV constexpr auto norm (pfc::complex <value_t> const & x) {
   return x.norm ();
}

template <typename value_t> CATTR_HOST_DEV constexpr auto & square (pfc::complex <value_t> & x) {
   return x.square ();
}

namespace literals {

CATTR_HOST_DEV constexpr inline auto operator "" _imag_f (long double const literal) {
   return pfc::complex <float> {0.0f, static_cast <float> (literal)};
}

CATTR_HOST_DEV constexpr inline auto operator "" _imag (unsigned long long const literal) {
   return pfc::complex <double> {0.0, static_cast <double> (literal)};
}

CATTR_HOST_DEV constexpr inline auto operator "" _imag (long double const literal) {
   return pfc::complex <double> {0.0, static_cast <double> (literal)};
}

CATTR_HOST_DEV constexpr inline auto operator "" _imag_l (long double const literal) {
   return pfc::complex <long double> {0.0l, literal};
}

CATTR_HOST_DEV constexpr inline auto operator "" _real_f (long double const literal) {
   return pfc::complex <float> {static_cast <float> (literal)};
}

CATTR_HOST_DEV constexpr inline auto operator "" _real (unsigned long long const literal) {
   return pfc::complex <double> {static_cast <double> (literal)};
}

CATTR_HOST_DEV constexpr inline auto operator "" _real (long double const literal) {
   return pfc::complex <double> {static_cast <double> (literal)};
}

CATTR_HOST_DEV constexpr inline auto operator "" _real_l (long double const literal) {
   return pfc::complex <long double> {literal};
}

// -------------------------------------------------------------------------------------------------

} }   // namespace pfc::literals

#endif   // PFC_COMPLEX_H
