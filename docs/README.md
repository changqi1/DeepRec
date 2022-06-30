## 背景介绍

本文从工程角度出发, 整理了相关资料, 简要剖析DeepRec架构实现, 框架层面的优化策略, 以及介绍以往落地的优化案例. 意在使后来者在相关的开发工作上, 有迹可循, 不必迷失在庞大的代码框架中.  DeepRec 以tensorflow v1.15 作为版本基础, 结合了Intel & NVIDIA在tensorflow上的众多优化, 以及Ali在超大规模的稀疏场景下的多年技术沉淀, 针对于稀疏模型在分布式/图优化/算子/Runtime等多方面进行了深度的性能优化工作. 详细的feature介绍可以参考如下:

- github: [https://github.com/alibaba/DeepRec](https://github.com/alibaba/DeepRec)
- docs: [https://deeprec.readthedocs.io/zh/latest/](https://deeprec.readthedocs.io/zh/latest/)

## 开发环境

### 代码结构

#### 克隆代码

#### 源码结构

### 工程编译

#### 环境准备

#### IDE

#### 构建

##### bazel介绍

- 编译组织方式
- select
- docs: https://docs.bazel.build/versions/0.26.0/user-manual.html

#### 验证

<br/>

<br/>

## 系统结构

DeepRec 已 C API 为分界线, 将整个系统划分为了前/后端两部分. 

1. **前端部分**: 主要为不同的编程语言提供相应编程接口, 负责构建计算图; 
2. **后端部分**: 提供运行时环境, 负责执行计算图;

从图操作的角度诠释DeepRec的工作流程如下:

<img src="pics/total_graph_operate.png" alt="" style="zoom:50%;" />

前端系统是一个支持多语言编程环境, 基于DeepRec的编程接口, 构造计算图. 在构图期间并不会真正执行相关的计算, 直到与后端建立Session, 并且将[Protobuf](https://developers.google.com/protocol-buffers)格式的GraphDef序列化传递给后端, 启动计算图的执行;

- 会话
- 计算图
- 算子

## 优化方法论

- 图优化
  - 常量折叠
  - 剪枝
- 算子优化 - (自定义算子(C++)-demo)
  - 提供pythonAPI
  - 图替换
- 图&算子优化 - (fusion-demo)

## 案例分析

1. Intel-tensorflow 优化工作介绍;
2. onednn 介绍;
3. 图优化;
4. mkl-related ops
5. embedding fusion 案例分析;
   1. 瓶颈分析;
   2. 改图;
   3. 添加自定义算子;
   4. 测试;

## 优化练习 

### 创建自定义算子

<br/>

<br/>

<br/>

### 算子优化

<br/>

### 图优化

<br/>

## 结语

<br/>

## DeepRec feature介绍
