//
// Copyright 2020-2021 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version Septmeber 2018)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#ifndef OPTIMIZERIF_H
#define OPTIMIZERIF_H

#ifdef KERNEL_MODULE
#include <linux/types.h>
#else
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#endif

#include "Common.h"
#include "MapMapToFloat.h"
#include "MapStringToFloat.h"
#include "MapStringToPtr.h"
#include "MapStringToString.h"
#include "MapVectorToMap.h"
#include "VectorFloat.h"
#include "VectorFloatToMap.h"
#include "VectorInt.h"
#include "VectorString.h"
#include "VectorVectorString.h"
#include "VectorPair.h"
#include "Suite.h"

#define FLOAT_PARAM 0

typedef enum Algorithm {
  EMPTY = 0, // default as null
  PSO = 1,   // partical swarm optimizer
  GA = 2,    // genetic algorithm optimizer
  DE = 3,    // differential evolution optimizer
  MOEAD = 4, // multi-object differential evolution optimizer
  BO = 5,    // bayesian optimizer
} Algorithm;

typedef struct ParamOptimizerIF {
  void (*update)(struct ParamOptimizerIF *self);
  void (*completeTrial)(struct ParamOptimizerIF *self, Map_StringToString param, Vector_Float result);
  bool_t (*getTrial)(struct ParamOptimizerIF *self, Map_StringToString param);
  Algorithm (*getAlgorithm)(struct ParamOptimizerIF *self);
#if FLOAT_PARAM
  void (*regist)(struct ParamOptimizerIF *self, char *key, float min, float max, int (*update)(char *s, float n));
#else
  void (*regist)(struct ParamOptimizerIF *self, char *key, int min, int max, int (*update)(char *s, int n));
#endif
  void (*unregist)(struct ParamOptimizerIF *self, char *key);
  void (*getOptimizedParam)(struct ParamOptimizerIF *self, Map_StringToString param);
  void (*getOptimizedParams)(struct ParamOptimizerIF *self, Map_VectorToMap param);
  char *(*getOptimizedTarget)(struct ParamOptimizerIF *self);
  Vector_Vector_String (*getOptimizedTargets)(struct ParamOptimizerIF *self);
  void (*getCurrentParam)(struct ParamOptimizerIF *self, Map_StringToString param);
  void (*calibrateParam)(struct ParamOptimizerIF *self, Map_StringToString param);
  //static struct ParamOptimizerIF *getParamOptimizer(OptParam<TA> &param);
  bool_t (*isTrainingEnd)(struct ParamOptimizerIF *self);
  void (*initLogging)(struct ParamOptimizerIF *self, char *argv);
  void (*setPCAWindow)(struct ParamOptimizerIF *self, int size);
} ParamOptimizerIF;

typedef struct OptParam {
  Algorithm algorithm;
  Suite *suite;
} OptParam;

typedef struct PSOOptParam {
  OptParam base;
  int iter_num;
  int swarm_num;
  float update_weight;
  float update_cp;
  float update_cg;
} PSOOptParam;

typedef struct GAOptParam {
  OptParam base;
  int gen_num;
  int pop_size;
  float mutp;
} GAOptParam;
typedef struct DEOptParam {
  OptParam base;
  int gen_num;
  int pop_size;
} DEOptParam;

typedef struct MOEADOptParam {
  OptParam base;
  int gen_num;
  int pop_size;
  int obj_num;
} MOEADOptParam;

typedef struct BOOptParam {
  OptParam base;
  int iter_num;
  int random_state;
  char *cfg;
} BOOptParam;

ParamOptimizerIF *getParamOptimizer(OptParam *p);
void ParamOptimizerIF_Dtor(ParamOptimizerIF *self);

int checkHelp(int argc, char *argv[]); //-1:error, >=0:no error, 0:exist other options, 1:only help
int tune(int argc, char *argv[], Suite *p_suite);
int getOptParam(Algorithm algo, Suite* p_suite, int gen, int pop, OptParam** pp_OptParam); 
int tuneSuiteWithOptParam(Suite *p_Suite, OptParam *p_OptParam);

#endif
