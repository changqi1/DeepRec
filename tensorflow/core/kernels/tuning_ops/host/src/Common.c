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

#include "Common.h"

int randomInt(int min, int max)
{
    return min + RAND_FUNC % (max - min + 1);
}

float randomFloat(float min, float max)
{
    float ret;
    //kernel_fpu_begin();
    ret = min + (max - min) * (float) RAND_FUNC / RAND_MAXV;
    //kernel_fpu_end();
    return ret;
}

#ifdef KERNEL_MODULE
double atof(const char* str)
{
    double s=0.0;
    double d=10.0;
    int jishu=0;
    bool_t flag=false_t;
    while (*str==' ') {
        str++;
    }
    if (*str=='-') {
        flag=true_t;
        str++;
    }
    if (!(*str>='0'&&*str<='9'))
        return s;
    while (*str>='0'&&*str<='9'&&*str!='.') {
        s=s*10.0+*str-'0';
        str++;
    }
    if(*str=='.')
        str++;
    while(*str>='0'&&*str<='9') {
        s=s+(*str-'0')/d;
        d*=10.0;
        str++;
    }
    if(*str=='e'||*str=='E') {
        str++;
        if(*str=='+') {
            str++;
            while(*str>='0'&&*str<='9') {
                jishu=jishu*10+*str-'0';
                str++;
            }
            while(jishu>0) {
                s*=10;
                jishu--;
            }
        }
        if(*str=='-') {
            str++;
            while(*str>='0'&&*str<='9') {
                jishu=jishu*10+*str-'0';
                str++;
            }
            while(jishu>0) {
                s/=10;
                jishu--;
            }
        }
    }
    return s*(flag? -1.0 : 1.0);
}

int atoi(const char* str) {
    int ret = 0;
    int sign = 1;
    if (str == NULL) {
        return 0;
    }
    while(*str == ' ')
        str++;
    if(*str == '-')
        sign = -1;
    if(*str == '-' || *str == '+')
        str++;
    while(*str >= '0' && *str <= '9') {
        ret = ret * 10 + (*str - '0');
        str++;
    }
    ret = sign * ret;
    return ret;
}

double fabs(double x) {
    if (x < 0.0) return -x;
    else return x;
}

float fabsf(float x) {
    if (x < 0.0) return -x;
    else return x;
}

double sqrt(double x) {
    double ret = 0.0;
    __asm__ __volatile__(
        "movups (%0),%%xmm1\n"
        "sqrtsd %%xmm1,%%xmm1\n"
        "movups %%xmm1,(%1)\n" : "=r" (ret) : "r" (x) );
    return ret;
}

float sqrtf(float x) {
    float ret = 0.0;
    __asm__ __volatile__(
        "movups (%0),%%xmm1\n"
        "sqrtss %%xmm1,%%xmm1\n"
        "movups %%xmm1,(%1)\n" : "=r" (ret) : "r" (x) );
    return ret;
}

double ceil(double x) {
    double ret = 0.0;
    __asm__ __volatile__(
        "movups (%0),%%xmm1\n"
        "roundsd $2,%%xmm1,%%xmm1\n"
        "movups %%xmm1,(%1)\n" : "=r" (ret) : "r" (x) );
    return ret;
}

double exp(double x) {
    return pow(2.71828182845904523536,x);
}

#ifndef USE_LIBM
double __kernel_standard(double x, double y, int type) {
    return 0.0;
}
int _dl_x86_cpu_features = 0xffffffff;
#define M_E 2.7182818284590452354       /* e */
extern double __ieee754_pow_sse2(double a,double b);
double __slowpow(double a,double b){return __ieee754_pow_sse2(a,b);}
double __slowexp(double a){return __ieee754_pow_sse2(M_E,a);}
double __ieee754_pow_fma4(double a,double b){return __ieee754_pow_sse2(a,b);}
double __ieee754_exp_fma4(double a){return __ieee754_pow_fma4(M_E,a);}
double __ieee754_exp_avx(double a){return __ieee754_pow_sse2(M_E,a);}

double pow(double a, double b)
{
    return __ieee754_pow_sse2(a,b);
}

#define M_PI 3.14159265358979323846     /* pi */
double __sin_sse2(double a);

double __mpcos (double __x, double __dx, char __range_reduce){return __sin_sse2(__x);}
double __mpcos1 (double __x, double __dx, char __range_reduce){return __sin_sse2(__x);}
double __mpsin (double __x, double __dx, char __range_reduce){return __sin_sse2(__x);}
double __mpsin1 (double __x, double __dx, char __range_reduce){return __sin_sse2(__x);}
double __sin_avx(double a) {return __sin_sse2(a);}
double __sin_fma4(double a) {return __sin_sse2(a);}
void __docos (double __x, double __dx, double __v[]){__sin_sse2(__x);}
void __dubsin (double __x, double __dx, double __v[]){__sin_sse2(__x);}
double __cos_avx(double a){return __sin_sse2(a);}
double __cos_fma4(double a){return __sin_sse2(a);}
int __branred (double x, double *a, double *aa){__sin_sse2(x);return 0;}
__thread int errno = 0;

double sin(double x) {
    return __sin_sse2(x);
}

double __ieee754_log_sse2(double a);

void __dbl_mp (double x, void *y, int p){__ieee754_log_sse2(x);}
void __mp_dbl (const void *x, double *y, int p){__ieee754_log_sse2(*y);}
void __mplog(char* str){__ieee754_log_sse2(0.0);}
void __add (const void *x, const void *y, void *z, int p){__ieee754_log_sse2(0.0);}
void __sub (const void *x, const void *y, void *z, int p){__ieee754_log_sse2(0.0);}
double __ieee754_log_fma4(double a){return __ieee754_log_sse2(a);}
double __ieee754_log_avx(double a){return __ieee754_log_sse2(a);}

double log(double x) {
    return __ieee754_log_sse2(x);
}

#endif

#endif
