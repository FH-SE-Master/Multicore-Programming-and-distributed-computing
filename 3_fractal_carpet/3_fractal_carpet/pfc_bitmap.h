//       $Id: pfc_bitmap.h 35947 2017-11-01 09:51:53Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/SE/MPV3/2016-WS/ILV/src/handouts/pfc_bitmap.h $
// $Revision: 35947 $
//     $Date: 2017-11-01 10:51:53 +0100 (Mi., 01 Nov 2017) $
//   Creator: peter.kulczycki<AT>fh-hagenberg.at
//   $Author: p20068 $
//
// Copyright: (c) 2016 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: Distributed under the Boost Software License, Version 1.0 (see
//            http://www.boost.org/LICENSE_1_0.txt).
//
//   Remarks: Currently, just 24-bit bitmaps are supported.

#if !defined PFC_BITMAP_H
#define      PFC_BITMAP_H

// version information -----------------------------------------------------------------------------

#undef PFC_BITMAP_VERSION
#undef PFC_BITMAP_VERSION_MAJOR
#undef PFC_BITMAP_VERSION_MINOR
#undef PFC_BITMAP_VERSION_PATCHLEVEL
#undef PFC_BITMAP_VERSION_REVISION

#define PFC_BITMAP_VERSION            "2.1.1"   /*                            */
#define PFC_BITMAP_VERSION_MAJOR      2         /* große Änderungen           */
#define PFC_BITMAP_VERSION_MINOR      1         /* funktionelle Erweiterungen */
#define PFC_BITMAP_VERSION_PATCHLEVEL 1         /* Fehlerbehebungen           */
#define PFC_BITMAP_VERSION_REVISION   33163     /* VCS Revisionsnummer        */

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// -------------------------------------------------------------------------------------------------

#if !defined PFC_MINI_HPP   /* header 'pfc-mini.hpp' not included */

#include <cstdint>

#undef  PFC_STATIC_ASSERT
#define PFC_STATIC_ASSERT(xpr) \
   static_assert ((xpr), "PFC: '" PFC_STRINGIZE (xpr) "'")

#undef  PFC_STRINGIZE
#define PFC_STRINGIZE(a) \
   #a

namespace pfc {

using byte_t  = std::uint8_t;  PFC_STATIC_ASSERT (sizeof (pfc::byte_t)  == 1);
using dword_t = std::uint32_t; PFC_STATIC_ASSERT (sizeof (pfc::dword_t) == 4);
using long_t  = std::int32_t;  PFC_STATIC_ASSERT (sizeof (pfc::long_t)  == 4);
using word_t  = std::uint16_t; PFC_STATIC_ASSERT (sizeof (pfc::word_t)  == 2);

template <typename T> inline bool is_lt (T const & a, T const & b) {
   return a < b;
}

template <typename T> inline bool is_eq (T const & a, T const & b) {
   return !pfc::is_lt (a, b) && !pfc::is_lt (b, a);
}

template <typename T> inline bool is_equal (T const & a, T const & b) {
   return a == b;
}

template <typename T> inline bool is_zero (T const & a) {
   static T const zero = T (); return pfc::is_eq (a, zero);   // a == zero
}
template <typename T> inline bool is_negative (T const & a) {
   static T const zero = T (); return pfc::is_lt (a, zero);
}

template <typename T> inline T abs (T const & a) {
   static T const zero = T (); return pfc::is_negative (a) ? zero - a : a;
}

template <typename T> inline T ceil_div (T const a, T const b) {
   PFC_STATIC_ASSERT (std::is_integral <T>::value);

   auto const m = pfc::abs (a) % b;

   if (pfc::is_zero (m)) {
      return a / b;

   } else if (pfc::is_negative (a)) {
      return (a + m) / b;

   } else {
      return (a + (b - m)) / b;
   }
}

template <typename T> inline T multiple_of (T const a, T const b) {
   PFC_STATIC_ASSERT (std::is_integral <T>::value);

   return pfc::ceil_div (a, b) * b;
}

template <typename T> inline bool cstr_empty (T * const p_cstr) {
   return (p_cstr == nullptr) || (*p_cstr == T (0));
}

template <typename T> inline T * mem_copy_ptr (T * const p_dst, T const * const p_src, int const size) {
   if ((p_dst != nullptr) && (p_src != nullptr) && (0 < size)) {
      std::memcpy (p_dst, p_src, size * sizeof (T));
   }

   return p_dst;
}

template <typename T> inline T * mem_set_ptr (T * const p_dst, int const c, int const size) {
   if ((p_dst != nullptr) && (0 < size)) {
      std::memset (p_dst, c, size * sizeof (T));
   }

   return p_dst;
}

template <typename T> inline T & mem_copy (T & dst, T const & src) {
   pfc::mem_copy_ptr (&dst, &src, 1); return dst;
}

template <typename T> inline T & mem_set (T & dst, int const c = 0) {
   pfc::mem_set_ptr (&dst, c, 1); return dst;
}

template <typename T> inline T & mem_reset (T & dst) {
   return pfc::mem_set (dst);
}

template <typename T> inline std::istream & read_ptr (std::istream & in, T * const ptr, int const count = 1) {
   if (in.good () && (ptr != nullptr) && (0 < count)) {
      in.read ((char *) ptr, count * sizeof (T));
   }

   return in;
}

template <typename T> inline std::istream & read (std::istream & in, T & val) {
   return pfc::read_ptr (in, &val, 1);
}

template <typename T> inline std::ostream & write_ptr (std::ostream & out, T const * const ptr, int const count = 1) {
   if (out.good () && (ptr != nullptr) && (0 < count)) {
      out.write ((char *) ptr, count * sizeof (T));
   }

   return out;
}

template <typename T> inline std::ostream & write (std::ostream & out, T const & val) {
   return pfc::write_ptr <T> (out, &val, 1);
}

}   // namespace pfc

#endif

// -------------------------------------------------------------------------------------------------

namespace pfc {

// byte-aligned types ------------------------------------------------------------------------------

#pragma pack (push, 1)

struct RGB_3_t {
   pfc::byte_t blue;
   pfc::byte_t green;
   pfc::byte_t red;
};

struct RGB_4_t {
   pfc::byte_t blue;
   pfc::byte_t green;
   pfc::byte_t red;
   pfc::byte_t reserved;
};

#pragma pack (pop)

PFC_STATIC_ASSERT (sizeof (pfc::RGB_3_t) == 3);
PFC_STATIC_ASSERT (sizeof (pfc::RGB_4_t) == 4);

// bitmap ------------------------------------------------------------------------------------------

class bitmap final {
   #pragma pack (push, 1)

   struct file_header_t {
      union {
         pfc::byte_t signature [2];   // file type; must be 'BM'
         pfc::word_t type;            // file type; must be 0x4d42
      };

      pfc::dword_t size;         // size, in bytes, of the bitmap file
      pfc::word_t  reserved_1;   // reserved; must be 0
      pfc::word_t  reserved_2;   // reserved; must be 0
      pfc::dword_t offset;       // offset, in bytes, from the beginning of the 'file_header_t' to the bitmap bits
   };

   struct info_header_t {
      pfc::dword_t size;            // number of bytes required by the structure
      pfc::long_t  width;           // width of the bitmap, in pixels
      pfc::long_t  height;          // height of the bitmap, in pixels
      pfc::word_t  planes;          // number of planes for the target device; must be 1
      pfc::word_t  bit_count;       // number of bits per pixel
      pfc::dword_t compression;     // type of compression; 0 for uncompressed RGB
      pfc::dword_t size_image;      // size, in bytes, of the image
      pfc::long_t  x_pels_pm;       // horizontal resolution, in pixels per meter
      pfc::long_t  y_pels_pm;       // vertical resolution, in pixels per meter
      pfc::dword_t clr_used;        // number of color indices in the color table
      pfc::dword_t clr_important;   // number of color indices that are considered important
   };

   #pragma pack (pop)

   public:
      using pixel_t      = pfc::RGB_4_t;
      using pixel_file_t = pfc::RGB_3_t;

      bitmap () {
         create (0, 0);
      }

      bitmap (int const width, int const height, pfc::byte_t * const p_image = nullptr) {
         create (width, height, p_image);
      }

      explicit bitmap (char const * const p_filename) {
         create (0, 0); from_file (p_filename);
      }

      explicit bitmap (std::string const & filename) {
         create (0, 0); from_file (filename);
      }

      explicit bitmap (std::istream & in) {
         create (0, 0); from_stream (in);
      }

      bitmap (bitmap const & src) {   // copy construction
         create (0, 0); copy (src);
      }

      bitmap (bitmap && src) {   // move construction
         create (0, 0); swap (src);
      }

      bitmap & operator = (bitmap const & rhs) {   // copy assignment
         if (&rhs != this) {
            bitmap tmp (rhs); swap (tmp);
         }

         return *this;
      }

      bitmap & operator = (bitmap && rhs) {   // move assignment
         swap (rhs); return *this;
      }

      void clear () {
         create (0, 0);
      }

      void create (int const width, int const height, pfc::byte_t * p_image = nullptr) {
         pfc::mem_reset (m_file_header);
         pfc::mem_reset (m_info_header);

         m_file_header.type       = 0x4d42;
         m_file_header.size       = static_cast <pfc::dword_t> (sizeof (file_header_t) + sizeof (info_header_t) + get_size_in_file (width, height));
         m_file_header.offset     = static_cast <pfc::dword_t> (sizeof (file_header_t) + sizeof (info_header_t));

         m_info_header.size       = sizeof (info_header_t);
         m_info_header.width      = static_cast <pfc::long_t> (width);
         m_info_header.height     = static_cast <pfc::long_t> (height);
         m_info_header.planes     =  1;
         m_info_header.bit_count  = 24;
         m_info_header.size_image = static_cast <pfc::dword_t> (get_size_in_file (width, height));

         pfc::mem_set_ptr (
            (m_p_image = (p_image == nullptr) ? pointer_pair <pfc::byte_t> (allocate <pfc::byte_t> (get_size (width, height)))
                                              : pointer_pair <pfc::byte_t> (p_image)).get (),
            0xff,
            get_size (width, height)
         );
      }

      bool from_file (char const * const p_filename) {
         bool ok {false};

         if (!pfc::cstr_empty (p_filename)) {
            std::ifstream in (p_filename, std::ios_base::binary); ok = from_stream (in);
         }

         return ok;
      }

      bool from_file (std::string const & filename) {
         return from_file (filename.c_str ());
      }

      bool from_stream (std::istream & in) {
         bool ok {false};

         if (in.good ()) {
            clear ();

            pfc::read (in, m_file_header);
            pfc::read (in, m_info_header);

            if (is_valid (m_file_header) && is_valid (m_info_header)) {
               m_p_image = pointer_pair <pfc::byte_t> (allocate <pfc::byte_t> (get_size ()));

               std::vector <pfc::byte_t> line  (get_stride_in_file (m_info_header.width));
               auto *                    p_dst (get_pixels ());

               for (int h {0}; h < get_height (); ++h) {
                  pfc::read_ptr (in, line.data (), static_cast <int> (line.size ())); auto const * p_src (line.data ());

                  for (int w {0}; w < get_width (); p_src += 3, ++p_dst, ++w) {
                     *((pixel_file_t *) p_dst) = *((pixel_file_t const *) p_src);
                  }
               }

               ok = in.good ();
            } else {
               clear ();
            }
         }

         return ok;
      }

      int get_height () const {
         return m_info_header.height;
      }

      pfc::byte_t * get_image () {
         return m_p_image.get ();
      }

      pfc::byte_t const * get_image () const {
         return m_p_image.get ();
      }

      pixel_t * get_pixels () {
         return reinterpret_cast <pixel_t *> (m_p_image.get ());
      }

      pixel_t const * get_pixels () const {
         return reinterpret_cast <pixel_t const *> (m_p_image.get ());
      }

      int get_size () const {
         return m_info_header.height * get_stride ();
      }

      int get_stride () const {
         return m_info_header.width * 4;
      }

      int get_width () const {
         return m_info_header.width;
      }

      void swap (bitmap & bmp) {
         if (&bmp != this) {
            std::swap (m_file_header, bmp.m_file_header);
            std::swap (m_info_header, bmp.m_info_header);
            std::swap (m_p_image,     bmp.m_p_image);
         }
      }

      bool to_file (char const * const p_filename) const {
         bool ok {false};

         if (!pfc::cstr_empty (p_filename)) {
            std::ofstream out (p_filename, std::ios_base::binary); ok = to_stream (out);
         }

         return ok;
      }

      bool to_file (std::string const & filename) const {
         return to_file (filename.c_str ());
      }

      bool to_stream (std::ostream & out) const {
         bool ok {false};

         if (out.good ()) {
            pfc::write (out, m_file_header);
            pfc::write (out, m_info_header);

            std::vector <pfc::byte_t> line  (get_stride_in_file (m_info_header.width));
            auto const *              p_src (get_pixels ());

            for (int h {0}; h < get_height (); ++h) {
               auto * p_dst (line.data ());

               for (int w {0}; w < get_width (); ++p_src, p_dst += 3, ++w) {
                  *((pixel_file_t *) p_dst) = *((pixel_file_t const *) p_src);
               }

               pfc::write_ptr (out, line.data (), static_cast <int> (line.size ()));
            }

            ok = out.good ();
         }

         return ok;
      }

   private:
      template <typename T> class pointer_pair final {
         public:
            pointer_pair () : m_ptr (nullptr), m_uptr (nullptr) {
            }

            explicit pointer_pair (T * const ptr) : m_ptr (ptr), m_uptr (nullptr) {
            }

            explicit pointer_pair (std::unique_ptr <T []> && uptr) : m_ptr (nullptr), m_uptr (std::move (uptr)) {
            }

            pointer_pair (pointer_pair const &) = delete;   // no copy construction

            pointer_pair (pointer_pair && src) : m_ptr (src.m_ptr), m_uptr (std::move (src.m_uptr)) {   // move construction
               src.m_ptr = nullptr;
            }

            pointer_pair & operator = (pointer_pair const &) = delete;   // no copy assignment

            pointer_pair & operator = (pointer_pair && rhs) {   // move assignment
               if (&rhs != this) {
                  m_ptr = rhs.m_ptr; m_uptr = std::move (rhs.m_uptr); rhs.m_ptr = nullptr;
               }

               return *this;
            }

            T * get () const {
               return (m_ptr != nullptr) ? m_ptr : m_uptr.get ();
            }

         private:
            T *                    m_ptr  {};
            std::unique_ptr <T []> m_uptr {};
      };

      template <typename T> static std::unique_ptr <pfc::byte_t []> allocate (int const size) {
         return std::unique_ptr <pfc::byte_t []> ((0 < size) ? new T [size] : nullptr);
      }

      static int get_size (int const width, int const height) {
         return height * get_stride (width);
      }

      static int get_size_in_file (int const width, int const height) {
         return height * get_stride_in_file (width);
      }

      static int get_stride (int const width) {
         return width * 4;
      }

      static int get_stride_in_file (int const width) {
         return pfc::multiple_of <int> (width * 3, 4);
      }

      static bool is_valid (file_header_t const & hdr) {
         bool const ok_offset     {hdr.offset     == sizeof (file_header_t) + sizeof (info_header_t)};
         bool const ok_reserved_1 {hdr.reserved_1 == 0};
         bool const ok_reserved_2 {hdr.reserved_2 == 0};
         bool const ok_size       {hdr.size       >= sizeof (file_header_t) + sizeof (info_header_t)};
         bool const ok_type       {hdr.type       == 0x4d42};

         return ok_offset && ok_reserved_1 && ok_reserved_2 && ok_size && ok_type;
      }

      static bool is_valid (info_header_t const & hdr) {
         bool const ok_bit_count     {hdr.bit_count                      == 24};
         bool const ok_clr_important {hdr.clr_important                  ==  0};
         bool const ok_clr_used      {hdr.clr_used                       ==  0};
         bool const ok_compression   {hdr.compression                    ==  0};
         bool const ok_planes        {hdr.planes                         ==  1};
         bool const ok_size          {hdr.size                           == sizeof (info_header_t)};
         bool const ok_size_image    {static_cast <int> (hdr.size_image) == get_size_in_file (hdr.width, hdr.height)};

         return ok_bit_count && ok_clr_important && ok_clr_used && ok_compression && ok_planes && ok_size && ok_size_image;
      }

      void copy (bitmap const & src) {
         if (&src != this) {
            pfc::mem_copy (m_file_header, src.m_file_header);
            pfc::mem_copy (m_info_header, src.m_info_header);

            pfc::mem_copy_ptr (
               (m_p_image = pointer_pair <pfc::byte_t> (allocate <pfc::byte_t> (src.get_size ()))).get (), src.m_p_image.get (), src.get_size ()
            );
         }
      }

      file_header_t              m_file_header {};
      info_header_t              m_info_header {};
      pointer_pair <pfc::byte_t> m_p_image     {};
};

// functions ---------------------------------------------------------------------------------------

inline void swap (pfc::bitmap & lhs, pfc::bitmap & rhs) {
   lhs.swap (rhs);
}

// -------------------------------------------------------------------------------------------------

}   // namespace pfc

#endif   // PFC_BITMAP_H
