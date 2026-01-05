#pragma once

#ifdef CONFIGDEFS
GEN_DEFINE(K2C_MAX_NDIM, 5)
#define K2C_STRUCTURE(NAME, MEMBERS) GEN_STRUCTURE(NAME: MEMBERS)
#else
#include <stddef.h>
#define K2C_MAX_NDIM 5
#define K2C_STRUCTURE(NAME, MEMBERS) struct NAME { MEMBERS; };
typedef struct k2c_tensor k2c_tensor;
#endif

#ifndef REALTIMEPROTOS
K2C_STRUCTURE(k2c_tensor,
  float * array;
  size_t ndim;
  size_t numel;
  size_t shape[K2C_MAX_NDIM])
#endif
