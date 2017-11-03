//       $Id: pfc_parallel.h 1306 2017-11-03 09:43:10Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/se/sw/mpv3/trunk/Lecture/Source/ACC/common/pfc_parallel.h $
// $Revision: 1306 $
//     $Date: 2017-11-03 10:43:10 +0100 (Fr., 03 Nov 2017) $
//   Creator: peter.kulczycki<AT>fh-hagenberg.at
//   $Author: p20068 $
//
// Copyright: (c) 2017 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: Distributed under the Boost Software License, Version 1.0 (see
//            http://www.boost.org/LICENSE_1_0.txt).

#if !defined PFC_PARALLEL
#define      PFC_PARALLEL

#include "./pfc_base.h"

#include <future>
#include <thread>
#include <vector>

namespace pfc {

// -------------------------------------------------------------------------------------------------

constexpr inline auto hardware_concurrency () noexcept {
   return std::max <int> (1, std::thread::hardware_concurrency ());
}

constexpr inline auto load_per_task (int const task, int const tasks, int const size) noexcept {
   return size / tasks + ((task < (size % tasks)) ? 1 : 0);
}

// -------------------------------------------------------------------------------------------------

class task_group final {
   public:
      explicit task_group () = default;

      task_group (task_group const &) = delete;
      task_group (task_group &&) = default;

     ~task_group () {
         join_all ();
      }

      task_group & operator = (task_group const &) = delete;
      task_group & operator = (task_group &&) = default;

      template <typename fun_t, typename ...args_t> void add (fun_t && fun, args_t && ...args) {
         m_group.push_back (
            std::async (std::launch::async, std::forward <fun_t> (fun), std::forward <args_t> (args)...)
         );
      }

      void join_all () {
         for (auto & f : m_group) f.wait ();
      }

   private:
      std::vector <std::future <void>> m_group;
};

// -------------------------------------------------------------------------------------------------

class thread_group final {
   public:
      explicit thread_group () = default;

      thread_group (thread_group const &) = delete;
      thread_group (thread_group &&) = default;

     ~thread_group () {
         join_all ();
      }

      thread_group & operator = (thread_group const &) = delete;
      thread_group & operator = (thread_group &&) = default;

      template <typename fun_t, typename ...args_t> void add (fun_t && fun, args_t && ...args) {
         m_group.emplace_back (std::forward <fun_t> (fun), std::forward <args_t> (args)...);
      }

      void join_all () {
         for (auto & t : m_group) if (t.joinable ()) t.join ();
      }

   private:
      std::vector <std::thread> m_group;
};

// -------------------------------------------------------------------------------------------------

template <typename fun_t> void parallel_range (pfc::task_group & group, int const tasks, int const size, fun_t && fun) {
   int begin {0};
   int end   {0};

   for (int t {0}; t < tasks; ++t) {
      end += pfc::load_per_task (t, tasks, size);

      if (end > begin) {
         group.add (std::forward <fun_t> (fun), t, begin, end);
      }

      begin = end;
   }
}

template <typename fun_t> void parallel_range (int const tasks, int const size, fun_t && fun) {
   pfc::task_group group; 
   pfc::parallel_range <fun_t> (group, tasks, size, std::forward <fun_t> (fun));
}

// -------------------------------------------------------------------------------------------------

inline bool set_priority_to_realtime () {
   #if defined PFC_WINDOWS_H_INCLUDED
      return SetPriorityClass (GetCurrentProcess (), REALTIME_PRIORITY_CLASS) != 0;
   #else
      return true;
   #endif
}

template <typename clock_t = std::chrono::steady_clock, typename duration_t = std::chrono::seconds> void warm_up_cpu (duration_t const how_long = duration_t {5}) {
   static_assert (clock_t::is_steady, "clock_t must denote a steady clock");

   auto              cores {std::max <int> (1, std::thread::hardware_concurrency ())};
   pfc::thread_group group {};

   while (0 < cores--) {
      group.add ([how_long, start = clock_t::now ()] {
         while ((clock_t::now () - start) < how_long);
      });
   }
}

// -------------------------------------------------------------------------------------------------

}   // namespace pfc

#endif   // PFC_PARALLEL
