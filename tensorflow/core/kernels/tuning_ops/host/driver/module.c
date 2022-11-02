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

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/miscdevice.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/poll.h>
#include <linux/list.h>
#include <linux/rbtree.h>
#include <linux/idr.h>
#include <linux/mutex.h>
#include <linux/kref.h>
#include <asm/uaccess.h>
#include "OptimizerIF.h"
#include "module.h"

EXPORT_SYMBOL(getParamOptimizer);
EXPORT_SYMBOL(ParamOptimizerIF_Dtor);

typedef struct ddwo_dev {
    struct miscdevice misc;
    struct mutex lock;
    struct idr opt_set;
} ddwo_dev, *ddwo_pdev;

static ddwo_pdev g_pdev = NULL;

typedef struct ddwo_list_param {
    struct hlist_node node;
    ddwo_pair_param param;
} ddwo_list_param, *ddwo_list_param_ptr;

#define DDWO_BUCKET_SIZE 10

static int string_to_hash(const char* str) {
    int ret = 0;
    if (NULL == str)
        return ret;
    while (*str) {
        ret = (ret << 5) + *str;
        str++;
    }
    return ret % DDWO_BUCKET_SIZE;
}

static int insert_pair(struct hlist_head header[], const char* name, ddwo_value_param_ptr val) {
    int h;
    ddwo_list_param_ptr item;
    struct hlist_node *iter;
    if (NULL == name || NULL == val || NULL == header) {
        return -EINVAL;
    }
    h = string_to_hash(name);
    hlist_for_each(iter, &(header[h])) {
        ddwo_list_param_ptr ptr = hlist_entry(iter, ddwo_list_param, node);
        if (strcmp(ptr->param.name.name, name) == 0) {
            ptr->param.value = *val;
            return 0;
        }
    }
    item = kmalloc(sizeof(ddwo_list_param), GFP_KERNEL);
    if (NULL == item) {
        printk(KERN_ERR "Fail to allocate node!\n");
        return -ENOMEM;
    }
    item->param.name.name_len = strlen(name) + 1;
    item->param.name.name = kmalloc(sizeof(char) * item->param.name.name_len, GFP_KERNEL);
    if (NULL == item->param.name.name) {
        printk(KERN_ERR "Fail to allocate item!\n");
        return -ENOMEM;
    }
    memcpy(item->param.name.name, name, sizeof(char) * item->param.name.name_len);
    item->param.value = *val;
    hlist_add_head(&(item->node), &(header[h]));
    return 0;
}

static ddwo_value_param_ptr get_pair(struct hlist_head header[], const char* name) {
    int h;
    struct hlist_node *iter;
    if (NULL == name || NULL == header) {
        return NULL;
    }
    h = string_to_hash(name);
    hlist_for_each(iter, &(header[h])) {
        ddwo_list_param_ptr ptr = hlist_entry(iter, ddwo_list_param, node);
        if (strcmp(ptr->param.name.name, name) == 0) {
            return &(ptr->param.value);
        }
    }
    return NULL;
}

static int delete_pair(struct hlist_head header[], const char* name) {
    int h;
    struct hlist_node *iter;
    if (NULL == name || NULL == header) {
        return -EINVAL;
    }
    h = string_to_hash(name);
    hlist_for_each(iter, &(header[h])) {
        ddwo_list_param_ptr ptr = hlist_entry(iter, ddwo_list_param, node);
        if (strcmp(ptr->param.name.name, name) == 0) {
            hlist_del(&(ptr->node));
            kfree(ptr->param.name.name);
            kfree(ptr);
            return 0;
        }
    }
    return 0;
}

static int clear_pairs(struct hlist_head header[]) {
    int h;
    if (NULL == header) {
        return -EINVAL;
    }
    for (h = 0; h < DDWO_BUCKET_SIZE; ++h) {
        while (!hlist_empty(&(header[h]))) {
            ddwo_list_param_ptr ptr = hlist_entry(header[h].first, ddwo_list_param, node);
            hlist_del(header[h].first);
            kfree(ptr->param.name.name);
            kfree(ptr);
        }
    }
    return 0;
}

static int set_pairs(ddwo_pair_param_ptr src, int len, struct hlist_head header[], ddwo_pair_param_ptr *dst, char*** names, int set) {
    ddwo_pair_param_ptr pairs;
    char ** ns;
    int ret, i;
    if (NULL == src || NULL == header || NULL == dst || NULL == names)
        return -EINVAL;
    if (0 == len) {
        *dst = NULL;
        *names = NULL;
        return 0;
    }
    pairs = kmalloc(sizeof(ddwo_pair_param) * len, GFP_KERNEL);
    if (NULL == pairs) {
        printk(KERN_ERR "Fail to alloc pairs!\n");
        *dst = NULL;
        *names = NULL;
        return -ENOMEM;
    }
    ns = kmalloc(sizeof(char*) * len, GFP_KERNEL);
    if (NULL == ns) {
        printk(KERN_ERR "Fail to alloc names!\n");
        *dst = NULL;
        *names = NULL;
        kfree(pairs);
        return -ENOMEM;
    }
    ret = copy_from_user(pairs, src, sizeof(ddwo_pair_param) * len);
    if (ret != 0) {
        printk(KERN_ERR "Fail to copy from user\n");
        *dst = NULL;
        *names = NULL;
        kfree(pairs);
        kfree(ns);
        return -EINVAL;
    }
    for (i = 0; i < len; ++i) {
        char* name = kmalloc(sizeof(char)*pairs[i].name.name_len, GFP_KERNEL);
        if (NULL == name) {
            printk(KERN_ERR "Fail to alloc name!\n");
            while((--i)<0) kfree(ns[i]);
            kfree(pairs);
            kfree(ns);
            *dst = NULL;
            *names = NULL;
            return -ENOMEM;
        }
        ret = copy_from_user(name, pairs[i].name.name, sizeof(char)*pairs[i].name.name_len);
        if (ret != 0) {
            printk(KERN_ERR "Fail to copy from user\n");
            while((--i)<0) kfree(ns[i]);
            kfree(pairs);
            kfree(ns);
            kfree(name);
            *dst = NULL;
            *names = NULL;
            return -EINVAL;
        }
        if (set) {
            ret = insert_pair(header, name, &(pairs[i].value));
            if (ret != 0) {
                printk(KERN_ERR "Fail to insert pairs\n");
                while((--i)<0) kfree(ns[i]);
                kfree(pairs);
                kfree(ns);
                kfree(name);
                *dst = NULL;
                *names = NULL;
                return ret;
            }
        }
        ns[i] = name;
    }
    *dst = pairs;
    *names = ns;
    return ret;
}

typedef struct ddwo_opt {
    struct kref refcount;
    int handle;
    struct mutex lock;
    ParamOptimizerIF* opt;
    struct hlist_head actions[DDWO_BUCKET_SIZE];
    struct hlist_head observers[DDWO_BUCKET_SIZE];
    struct hlist_head targets[DDWO_BUCKET_SIZE];
} ddwo_opt, *ddwo_popt;

static ddwo_popt ddwo_init_opt(ParamOptimizerIF* opt);
static void ddwo_release_opt(struct kref *ref);
static void ddwo_deinit_opt(ddwo_popt popt);
static int ddwo_open(struct inode *inode, struct file *filep);
static int ddwo_release(struct inode *inode, struct file *filep);
static long ddwo_ioctl(struct file *filep,unsigned int cmd,unsigned long arg);

static const struct file_operations ddwo_fops = {
    .owner = THIS_MODULE,
    .open = ddwo_open,
    .release = ddwo_release,
    .unlocked_ioctl = ddwo_ioctl,
};

static struct miscdevice ddwo_misc = {
    .minor = MISC_DYNAMIC_MINOR,
    .name = "ddwo",
    .fops = &ddwo_fops,
};

static ddwo_popt ddwo_init_opt(ParamOptimizerIF* opt) {
    ddwo_popt ret = kmalloc(sizeof(struct ddwo_opt), GFP_KERNEL);
    if (ret != NULL) {
        int i;
        for (i = 0; i < DDWO_BUCKET_SIZE; ++i) {
            INIT_HLIST_HEAD(&(ret->actions[i]));
            INIT_HLIST_HEAD(&(ret->observers[i]));
            INIT_HLIST_HEAD(&(ret->targets[i]));
        }
        ret->opt = opt;
        mutex_lock(&(g_pdev->lock));
        ret->handle = idr_alloc(&(g_pdev->opt_set), ret, 0, 0, GFP_KERNEL);
        mutex_unlock(&(g_pdev->lock));
        if (ret->handle < 0) {
            ddwo_release_opt(&(ret->refcount));
            printk(KERN_ERR "Fail to alloc idr!\n");
            return NULL;
        }
        mutex_init(&(ret->lock));
        kref_init(&(ret->refcount));
    }
    return ret;
}

static void ddwo_release_opt(struct kref *ref) {
    ddwo_popt popt = container_of(ref, ddwo_opt, refcount);
    if (popt->opt != NULL)
        ParamOptimizerIF_Dtor(popt->opt);
    clear_pairs(popt->actions);
    clear_pairs(popt->observers);
    clear_pairs(popt->targets);
    if (popt->handle > 0) {
        mutex_lock(&(g_pdev->lock));
        idr_remove(&(g_pdev->opt_set), popt->handle);
        mutex_unlock(&(g_pdev->lock));
    }
    mutex_destroy(&(popt->lock));
    kfree(popt);
}

static void ddwo_deinit_opt(ddwo_popt popt) {
    if (popt != NULL) {
        kref_put(&(popt->refcount), ddwo_release_opt);
    }
}

static int ddwo_open(struct inode *inode, struct file *filep) {
    filep->private_data = NULL;
    return 0;
}

static int ddwo_release(struct inode *inode, struct file *filep) {
    ddwo_popt opt = (ddwo_popt)(filep->private_data);
    if (opt != NULL) {
        ddwo_deinit_opt(opt);
        filep->private_data = NULL;
    }
    return 0;
}

static long ddwo_ioctl(struct file *filep,unsigned int cmd,unsigned long arg) {
    int ret = 0;
    ddwo_popt po = (ddwo_popt)(filep->private_data);
    //if (NULL == po) return -ENOMEM;
    switch (cmd) {
        case DDWO_IOCTL_CREATE_CONTEXT:
            {
                ParamOptimizerIF* opt;
                ddwo_ioctl_opt_param param;
                ret = copy_from_user(&param, (unsigned char*)arg, sizeof(ddwo_ioctl_opt_param));
                if (ret != 0) {
                    printk(KERN_ERR "Fail to copy from user\n");
                    return -EINVAL;
                } else {
                    printk(KERN_DEBUG "Got request for algorithm %d!\n", (int)(param.algorithm));
                }
                if (po != NULL) {
                    printk(KERN_ERR "Unsupport multi context!");
                    return -EINVAL;
                }
                if (param.handle > 0) {
                    mutex_lock(&(g_pdev->lock));
                    po = idr_find(&(g_pdev->opt_set), param.handle);
                    mutex_unlock(&(g_pdev->lock));
                    if (NULL == po) {
                        printk(KERN_ERR "Not found handle!\n");
                        return -EINVAL;
                    }
                    kref_get(&(po->refcount));
                } else {
                    switch (param.algorithm) {
                        case PSO:
                            param.param.pso.base.suite = NULL;
                            opt = getParamOptimizer(&param.param.pso.base);
                            if (NULL == filep->private_data) {
                                printk(KERN_ERR "Fail to create pso param optimizer!\n");
                                return -EINVAL;
                            }
                            break;
                        case GA:
                            param.param.ga.base.suite = NULL;
                            opt = getParamOptimizer(&param.param.ga.base);
                            if (NULL == filep->private_data) {
                                printk(KERN_ERR "Fail to create ga param optimizer!\n");  
                                return -EINVAL;
                            }
                            break;
                        case BO:
                            param.param.bo.base.suite = NULL;
                            opt = getParamOptimizer(&param.param.bo.base);
                            if (NULL == filep->private_data) {
                                printk(KERN_ERR "Fail to create bo param optimizer!\n");
                                return -EINVAL;
                            }
                            break;
                        case DE:
                            param.param.bo.base.suite = NULL;
                            opt = getParamOptimizer(&param.param.de.base);
                            if (NULL == filep->private_data) {
                                printk(KERN_ERR "Fail to create de param optimizer!\n");
                                return -EINVAL;
                            }
                            break;
                        case MOEAD:
                            param.param.moead.base.suite = NULL;
                            opt = getParamOptimizer(&param.param.moead.base);
                            if (NULL == filep->private_data) {
                                printk(KERN_ERR "Fail to create moead param optimizer!\n");
                                return -EINVAL;
                            }
                            break;
                        default:
                            return -EINVAL;
                    }
                    po = ddwo_init_opt(opt);
                    if (NULL == po) return -ENOMEM;
                    ret = copy_to_user((unsigned char*)arg, &param, sizeof(ddwo_ioctl_opt_param));
                }
                filep->private_data = po;
            }
            break;
        case DDWO_IOCTL_REGIST_PARAM:
            {
                ddwo_ioctl_regist_param param;
                char * name;
                ret = copy_from_user(&param, (unsigned char*)arg, sizeof(ddwo_ioctl_regist_param));
                if (ret != 0) {
                    printk(KERN_ERR "Fail to copy from user\n");
                    return -EINVAL;
                }
                if (param.name.name_len <= 0) {
                    printk(KERN_ERR "Invalid key name!\n");
                    return -EINVAL;
                }
                name = kmalloc(sizeof(char) * param.name.name_len, GFP_KERNEL);
                if (NULL == name) {
                    printk(KERN_ERR "Fail to alloc string!\n");
                    return -ENOMEM;
                }
                ret = copy_from_user(name, param.name.name, sizeof(char) * param.name.name_len);
                if (ret != 0) {
                    printk(KERN_ERR "Fail to copy from user\n");
                    kfree(name);
                    return -EINVAL;
                }
                if (NULL == po->opt) {
                    printk(KERN_ERR "No context for the device!\n");
                    kfree(name);
                    return -EINVAL;
                }
                mutex_lock(&(po->lock));
                switch (param.min.value_type) {
                    case 0:
                        po->opt->regist(po->opt, name, param.min.value.int_val, param.max.value.int_val, NULL);
                        break;
                    case 1:
                        po->opt->regist(po->opt, name, param.min.value.float_val, param.max.value.float_val, NULL);
                        break;
                    default:
                        printk(KERN_ERR "Unsupported key type!\n");
                        ret = -EINVAL;
                        break;
                }
                if (0 == ret && 0 != insert_pair(po->actions, name, &(param.min))) {
                    printk(KERN_ERR "Fail to insert pair!\n");
                    ret = -EINVAL;
                }
                mutex_unlock(&(po->lock));
                kfree(name);
            }
            break;
        case DDWO_IOCTL_UNREGIST_PARAM:
            {
                ddwo_ioctl_regist_param param;
                char * name;
                ret = copy_from_user(&param, (unsigned char*)arg, sizeof(ddwo_ioctl_regist_param));
                if (ret != 0) {
                    printk(KERN_ERR "Fail to copy from user\n");
                    return -EINVAL;
                }
                if (param.name.name_len <= 0) {
                    printk(KERN_ERR "Invalid key name!\n");
                    return -EINVAL;
                }
                name = kmalloc(sizeof(char) * param.name.name_len, GFP_KERNEL);
                if (NULL == name) {
                    printk(KERN_ERR "Fail to alloc string!\n");
                    return -ENOMEM;
                }
                ret = copy_from_user(name, param.name.name, sizeof(char) * param.name.name_len);
                if (ret != 0) {
                    printk(KERN_ERR "Fail to copy from user\n");
                    kfree(name);
                    return -EINVAL;
                }
                if (NULL == po->opt) {
                    printk(KERN_ERR "No context for the device!\n");
                    kfree(name);
                    return -EINVAL;
                }
                mutex_lock(&(po->lock));
                po->opt->unregist(po->opt, name);
                if (0 == ret && 0 != delete_pair(po->actions, name)) {
                    printk(KERN_ERR "Fail to delete pair!\n");
                    ret = -EINVAL;
                }
                mutex_unlock(&(po->lock));
                kfree(name);
            }
            break;
        case DDWO_IOCTL_UPDATE:
            {
                ddwo_ioctl_update_param param;
                ddwo_pair_param_ptr pairs = NULL;
                ddwo_pair_param_ptr targets = NULL;
                ddwo_pair_param_ptr observers = NULL;
                char** pairs_name = NULL;
                char** targets_name = NULL;
                char** observers_name = NULL;
                int i;
                ret = copy_from_user(&param, (unsigned char*)arg, sizeof(ddwo_ioctl_update_param));
                if (ret != 0) {
                    printk(KERN_ERR "Fail to copy from user\n");
                    return -EINVAL;
                }
                mutex_lock(&(po->lock));
                if (param.op_set || param.op_get_opt || param.op_get) {
                    ret = set_pairs(param.pairs, param.pairs_len, po->actions, &pairs, &pairs_name, param.op_set);
                    if (ret != 0) {
                        mutex_unlock(&(po->lock));
                        printk(KERN_ERR "Fail to set pairs\n");
                        if (pairs != NULL) {
                            for (i = 0; i < param.pairs_len; ++i) kfree(pairs_name[i]);
                            kfree(pairs_name);
                            kfree(pairs);
                        }
                        return ret;
                    }
                    ret = set_pairs(param.targets, param.targets_len, po->targets, &targets, &targets_name, param.op_set);
                    if (ret != 0) {
                        mutex_unlock(&(po->lock));
                        printk(KERN_ERR "Fail to set targets\n");
                        if (pairs != NULL) {
                            for (i = 0; i < param.pairs_len; ++i) kfree(pairs_name[i]);
                            kfree(pairs_name);
                            kfree(pairs);
                        }
                        if (targets != NULL) {
                            for (i = 0; i < param.targets_len; ++i) kfree(targets_name[i]);
                            kfree(targets_name);
                            kfree(targets);
                        }
                        return ret;
                    }
                    ret = set_pairs(param.observers, param.observers_len, po->observers, &observers, &observers_name, param.op_set);
                    if (ret != 0) {
                        mutex_unlock(&(po->lock));
                        printk(KERN_ERR "Fail to set observers\n");
                        if (pairs != NULL) {
                            for (i = 0; i < param.pairs_len; ++i) kfree(pairs_name[i]);
                            kfree(pairs_name);
                            kfree(pairs);
                        }
                        if (targets != NULL) {
                            for (i = 0; i < param.targets_len; ++i) kfree(targets_name[i]);
                            kfree(targets_name);
                            kfree(targets);
                        }
                        if (observers != NULL) {
                            for (i = 0; i < param.observers_len; ++i) kfree(observers_name[i]);
                            kfree(observers_name);
                            kfree(observers);
                        }
                        return ret;
                    }
                }
                if (param.op_update) {
                    po->opt->update(po->opt);
                }
                if (param.op_get || param.op_get_opt) {
                    Map_StringToString map = NULL;
                    const char* str = NULL;
                    if (param.op_get_opt) {
                        map = Map_StringToString_Ctor();
                        if (map != NULL)
                            po->opt->getOptimizedParam(po->opt, map);
                        str = po->opt->getOptimizedTarget(po->opt);
                    }
                    if (pairs != NULL) {
                        for (i = 0; i < param.pairs_len; ++i) {
                            if (param.op_get) {
                                ddwo_value_param_ptr ptr = get_pair(po->actions, pairs_name[i]);
                                if (ptr != NULL) param.pairs[i].value = *ptr;
                            }
                            if (param.op_get_opt && map != NULL) {
                                Pair_StringToString* p = Map_StringToString_Find(map, pairs_name[i]);
                                if (p != NULL) {
                                    switch (param.pairs[i].value.value_type) {
                                        case 0:
                                            param.pairs[i].value.value.int_val = atoi(p->m_value);
                                            break;
                                        case 1:
                                            param.pairs[i].value.value.float_val = atof(p->m_value);
                                            break;
                                        default:
                                            break;
                                    }
                                }
                            }
                        }
                        if (param.pairs_len > 0) {
                            ret = copy_to_user(param.pairs, pairs, sizeof(ddwo_pair_param) * param.pairs_len);
                            if (ret != 0) printk(KERN_ERR "Fail to copy to user for pairs\n");
                        }
                    }
                    if (0 == ret && targets != NULL) {
                        for (i = 0; i < param.targets_len; ++i) {
                            if (param.op_get) {
                                ddwo_value_param_ptr ptr = get_pair(po->targets, targets_name[i]);
                                if (ptr != NULL) param.targets[i].value = *ptr;
                            }
                            if (param.op_get_opt && str != NULL) {
                                switch (param.targets[i].value.value_type) {
                                    case 0:
                                        param.targets[i].value.value.int_val = atoi(str);
                                        break;
                                    case 1:
                                        param.targets[i].value.value.float_val = atof(str);
                                        break;
                                    default:
                                        break;
                                }
                            }
                        }
                        if (param.targets_len > 0) {
                            ret = copy_to_user(param.targets, targets, sizeof(ddwo_pair_param) * param.targets_len);
                            if (ret != 0) printk(KERN_ERR "Fail to copy to user for targets\n");
                        }
                    }
                    if (0 == ret && observers != NULL) {
                        for (i = 0; i < param.observers_len; ++i) {
                            ddwo_value_param_ptr ptr = get_pair(po->observers, observers[i].name.name);
                            if (ptr != NULL) param.observers[i].value = *ptr;
                        }
                        if (param.observers_len > 0) {
                            ret = copy_to_user(param.observers, observers, sizeof(ddwo_pair_param) * param.observers_len);
                            if (ret != 0) printk(KERN_ERR "Fail to copy to user for observers\n");
                        }
                    }
                    if (map != NULL) Map_StringToString_Dtor(map);
                }
                mutex_unlock(&(po->lock));
                if (pairs != NULL) {
                    for (i = 0; i < param.pairs_len; ++i) kfree(pairs_name[i]);
                    kfree(pairs_name);
                    kfree(pairs);
                }
                if (targets != NULL) {
                    for (i = 0; i < param.targets_len; ++i) kfree(targets_name[i]);
                    kfree(targets_name);
                    kfree(targets);
                }
                if (observers != NULL) {
                    for (i = 0; i < param.observers_len; ++i) kfree(observers_name[i]);
                    kfree(observers_name);
                    kfree(observers);
                }
            }
            break;
        case DDWO_IOCTL_GET_ALGORITHM:
            {
                ddwo_ioctl_get_algorithm param;
                mutex_lock(&(po->lock));
                param.algorithm = po->opt->getAlgorithm(po->opt);
                mutex_unlock(&(po->lock));
                ret = copy_to_user((unsigned char*)arg, &param, sizeof(ddwo_ioctl_get_algorithm));
                if (ret != 0)
                    printk(KERN_ERR "Fail to copy to user for algorithm!\n");
            }
            break;
        case DDWO_IOCTL_IS_END:
            {
                ddwo_ioctl_is_end param;
                mutex_lock(&(po->lock));
                param.end = po->opt->isTrainingEnd(po->opt);
                mutex_unlock(&(po->lock));
                ret = copy_to_user((unsigned char*)arg, &param, sizeof(ddwo_ioctl_is_end));
                if (ret != 0)
                    printk(KERN_ERR "Fail to copy to user for end!\n");
            }
            break;
        case DDWO_IOCTL_PCA_WINDOW:
            {
                ddwo_ioctl_pca_window param;
                ret = copy_from_user(&param, (unsigned char*)arg, sizeof(ddwo_ioctl_pca_window));
                if (ret != 0) {
                    printk(KERN_ERR "Fail to copy from user for pca window!\n");
                } else {
                    mutex_lock(&(po->lock));
                    po->opt->setPCAWindow(po->opt, param.window_size);
                    mutex_unlock(&(po->lock));
                }
            }
            break;
        default:
            return -EINVAL;
    }
    return ret;
}

static int __init ddwo_init(void) {
    int ret;
    ddwo_pdev pdev = kmalloc(sizeof(struct ddwo_dev), GFP_KERNEL);
    if (NULL == pdev) {
        printk(KERN_ERR "Fail to create ddwo device!");
        return -ENOMEM;
    }
    pdev->misc = ddwo_misc;
    mutex_init(&(pdev->lock));
    idr_init(&(pdev->opt_set));
    ret = misc_register(&(pdev->misc));
    if (ret != 0) {
        printk(KERN_ERR "Fail to register ddwo device!");
        return ret;
    }
    g_pdev = pdev;
    printk(KERN_DEBUG "DDWO module has been loaded.\n");
    return ret;
}

static void __exit ddwo_exit(void) {
    ddwo_pdev pdev = g_pdev;
    if (NULL != pdev) {
        struct idr *idp = &(pdev->opt_set);
        ddwo_popt popt = NULL;
        int handle = -1;
        g_pdev = NULL;
        misc_deregister(&(pdev->misc));
        idr_for_each_entry(idp, popt, handle) {
            if (kref_put(&(popt->refcount), ddwo_release_opt) != 1)
                printk(KERN_ERR "Opt %p is still alive with handle %d!\n", popt, handle);
        }
        idr_destroy(&(pdev->opt_set));
        mutex_destroy(&(pdev->lock));
        kfree(pdev);
    }
    printk(KERN_DEBUG "DDWO module has been unloaded.\n");
}

module_init(ddwo_init);
module_exit(ddwo_exit);

MODULE_AUTHOR("jiangming.wu@intel.com>");
MODULE_AUTHOR("yanjie.pan@intel.com");
MODULE_AUTHOR("yulong.li@intel.com");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Intel DDWO driver");

