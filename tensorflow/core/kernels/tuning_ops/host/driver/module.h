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

#ifndef _DDWO_MODULE_H
#define _DDWO_MODULE_H

#include <linux/ioctl.h>
#include "OptimizerIF.h"

typedef struct ddwo_ioctl_opt_param {
    Algorithm algorithm;
    int handle; // if create new context, it should be -1; otherwise, it should be handle for other context; it will be replaced with real handle finally.
    union opt_param {
        PSOOptParam pso;
        GAOptParam ga;
        BOOptParam bo;
        DEOptParam de;
        MOEADOptParam moead;
    } param;
} ddwo_ioctl_opt_param;

typedef struct ddwo_value_param {
    int value_type; // 0 for int, 1 for float
    union {
        int int_val;
        float float_val;
    } value;
} ddwo_value_param, *ddwo_value_param_ptr;

typedef struct ddwo_name_param {
    char* name;
    int name_len;
} ddwo_name_param, *ddwo_name_param_ptr;

typedef struct ddwo_pair_param {
    ddwo_name_param name;
    ddwo_value_param value;
} ddwo_pair_param, *ddwo_pair_param_ptr;

typedef struct ddwo_ioctl_regist_param {
    ddwo_name_param name;
    ddwo_value_param min;
    ddwo_value_param max;
} ddwo_ioctl_regist_param;

typedef struct ddwo_ioctl_unregist_param {
    ddwo_name_param name;
} ddwo_ioctl_unregist_param;

typedef struct ddwo_ioctl_update_param {
    int op_set : 1;
    int op_get : 1;
    int op_update : 1;
    int op_get_opt : 1;
    int pairs_len;
    int targets_len;
    int observers_len;
    ddwo_pair_param_ptr pairs;
    ddwo_pair_param_ptr targets;
    ddwo_pair_param_ptr observers;
} ddwo_ioctl_update_param;

typedef struct ddwo_ioctl_get_algorithm {
    Algorithm algorithm;
} ddwo_ioctl_get_algorithm;

typedef struct ddwo_ioctl_is_end {
    int end;
} ddwo_ioctl_is_end;

typedef struct ddwo_ioctl_pca_window {
    int window_size;
} ddwo_ioctl_pca_window;

#define DDWO_IOCTL_OPT 0xFF

#define DDWO_IOCTL_CREATE_CONTEXT \
    _IOWR(DDWO_IOCTL_OPT, 0, ddwo_ioctl_opt_param)

#define DDWO_IOCTL_REGIST_PARAM \
    _IOW(DDWO_IOCTL_OPT, 1, ddwo_ioctl_regist_param)

#define DDWO_IOCTL_UNREGIST_PARAM \
    _IOW(DDWO_IOCTL_OPT, 2, ddwo_ioctl_unregist_param)

#define DDWO_IOCTL_UPDATE \
    _IOWR(DDWO_IOCTL_OPT, 3, ddwo_ioctl_update_param)

#define DDWO_IOCTL_GET_ALGORITHM \
    _IOR(DDWO_IOCTL_OPT, 4, ddwo_ioctl_get_algorithm)

#define DDWO_IOCTL_IS_END \
    _IOR(DDWO_IOCTL_OPT, 5, ddwo_ioctl_is_end)

#define DDWO_IOCTL_PCA_WINDOW \
    _IOW(DDWO_IOCTL_OPT, 6, ddwo_ioctl_pca_window)

#endif

