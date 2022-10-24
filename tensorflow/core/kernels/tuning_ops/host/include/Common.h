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

#ifndef COMMON_H
#define COMMON_H

#define nullptr NULL

#ifdef KERNEL_MODULE
typedef int bool_t;
#define true_t 1
#define false_t 0
#include <linux/types.h>
#include <linux/slab.h>
#define MALLOC(s) kmalloc(s, GFP_KERNEL)
#define FREE kfree
#define PRINTF printk
extern unsigned int prandom_u32(void);
#define RAND_FUNC (prandom_u32()>>24)
#define RAND_MAXV 0xff
double pow(double a, double b);
double atof(const char* str);
float fabsf(float x);
double fabs(double x);
double log(double x);
double sin(double x);
double ceil(double x);
double sqrt(double x);
float sqrtf(float x);
double exp(double x);
int atoi(const char* str);
#else
#include <stdbool.h>
typedef bool bool_t;
#define true_t true
#define false_t false
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#define MALLOC(s) malloc(s)
#define FREE free
#define PRINTF printf
#define RAND_FUNC rand()
#define RAND_MAXV RAND_MAX
#endif

int randomInt(int min, int max);
float randomFloat(float min, float max);

#define SWAPInt(A, B) \
    int temp = A; \
    A = B; \
    B = temp;

#define SWAPFloat(A, B) \
    float temp = A; \
    A = B; \
    B = temp;

#endif
