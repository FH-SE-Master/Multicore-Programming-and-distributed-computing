//       $Id: pfc_windows.h 1231 2017-10-12 18:46:26Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/se/sw/mpv3/trunk/Lecture/Source/ACC/common/pfc_windows.h $
// $Revision: 1231 $
//     $Date: 2017-10-12 20:46:26 +0200 (Do., 12 Okt 2017) $
//   Creator: peter.kulczycki<AT>fh-hagenberg.at
//   $Author: p20068 $
//
// Copyright: (c) 2017 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: Distributed under the Boost Software License, Version 1.0 (see
//            http://www.boost.org/LICENSE_1_0.txt).

#if !defined PFC_WINDOWS_H
#define      PFC_WINDOWS_H

// -------------------------------------------------------------------------------------------------

#undef  NOMINMAX
#define NOMINMAX

#undef  STRICT
#define STRICT

#undef  VC_EXTRALEAN
#define VC_EXTRALEAN

#undef  WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN

#include <windows.h>

#undef  PFC_WINDOWS_H_INCLUDED
#define PFC_WINDOWS_H_INCLUDED

// -------------------------------------------------------------------------------------------------

#endif   // PFC_WINDOWS_H
