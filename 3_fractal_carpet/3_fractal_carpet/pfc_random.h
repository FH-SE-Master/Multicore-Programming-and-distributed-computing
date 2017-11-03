//       $Id: pfc_random.h 35947 2017-11-01 09:51:53Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/SE/MPV3/2016-WS/ILV/src/handouts/pfc_random.h $
// $Revision: 35947 $
//     $Date: 2017-11-01 10:51:53 +0100 (Mi., 01 Nov 2017) $
//   Creator: peter.kulczycki<AT>fh-hagenberg.at
//   $Author: p20068 $
//
// Copyright: (c) 2017 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: Distributed under the Boost Software License, Version 1.0 (see
//            http://www.boost.org/LICENSE_1_0.txt).

#if !defined PFC_RANDOM_H
#define      PFC_RANDOM_H

#include "./pfc_traits.h"

#include <algorithm>
#include <array>
#include <functional>
#include <random>

namespace pfc {

// -------------------------------------------------------------------------------------------------

using default_random_engine_t = std::mt19937_64;

template <typename E = pfc::default_random_engine_t> class random final {
   public:
      using engine_t = E;

      random () = default;

      random (random const &) = delete;
      random (random &&) = delete;

      random & operator = (random const &) = delete;
      random & operator = (random &&) = delete;

      template <typename value_t> value_t get_random_uniform (pfc::integral <value_t> const from, value_t const to) {
         return std::uniform_int_distribution <value_t> {from, to} (m_engine);
      }

      template <typename value_t> value_t get_random_uniform (pfc::floating_point <value_t> const from, value_t const to) {
         return std::uniform_real_distribution <value_t> {from, to} (m_engine);
      }

   private:
      static auto get_engine () {
         using                 result_type = typename engine_t::result_type;
         static constexpr auto state_size  = engine_t::state_size;

         static std::random_device                   random_device;
         static std::array <result_type, state_size> seed_values;

         std::generate (std::begin (seed_values), std::end (seed_values), std::ref (random_device));

         std::seed_seq seed_sequence (std::begin (seed_values), std::end (seed_values));

         return engine_t {seed_sequence};
      }

      engine_t m_engine {get_engine ()};
};

// -------------------------------------------------------------------------------------------------

template <typename engine_t = pfc::default_random_engine_t, typename value_t> value_t get_random_uniform (value_t const from, value_t const to) {
   static pfc::random <engine_t> rnd; return rnd.get_random_uniform (from, to);
}

// -------------------------------------------------------------------------------------------------

}   // namespace pfc

#endif   // PFC_RANDOM_H
