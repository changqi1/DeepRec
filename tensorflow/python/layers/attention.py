# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import layers as keras_layers
from tensorflow.python.layers import base
from tensorflow.python.layers import core as core_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.util.tf_export import tf_export


# TODO:add shape fusion
def _fused_QKV_dense(inputs, dim):
  with ops.name_scope('_fused_QKV_dense'):
    Query = core_layers.dense(inputs, dim, activation=nn.relu, name='Query')
    Key = core_layers.dense(inputs, dim, activation=nn.relu, name='Key')
    Value = core_layers.dense(inputs, dim, activation=nn.relu, name='Value')
  return Query, Key, Value


def _fused_KV_dense(inputs, dim):
  with ops.name_scope('_fused_KV_dense'):
    Key = core_layers.dense(inputs, dim, activation=nn.relu, name='Key')
    Value = core_layers.dense(inputs, dim, activation=nn.relu, name='Value')
  return Key, Value


def _fused_split_concat(inputs, split_nums, split_axis, concat_axis):
  with ops.name_scope('_fused_conact_split'):
    outputs = array_ops.split(inputs, split_nums, axis=split_axis)
    outputs = array_ops.concat(outputs, axis=concat_axis)
  return outputs


class Attention(keras_layers.Attention, base.Layer):
  def __init__(self, causal=False, use_scale=None, trainable=True, name=None, **kwargs):
    super(Attention, self).__init__(causal=False,
                                    use_scale=None,
                                    trainable=trainable,
                                    name=name,
                                    **kwargs)


# @tf_export(v1=['layers.attention'])
def attention(inputs, mask, causal=False, use_scale=None, name=None, reuse=None):
  '''Dot-product attention layer, a.k.a. Luong-style attention.

    Inputs are `Query` tensor of shape `[batch_size, Tq, dim]`, `Key` tensor of
    shape `[batch_size, Tk, dim]` and `Value` tensor of shape
    `[batch_size, Tk, dim]`. The calculation follows the steps:

    1. Calculate scores with shape `[batch_size, Tq, Tk]` as a `query`-`key` dot
      product: `scores = tf.matmul(Q, K, transpose_b=True)`.
    2. If use scale, a scalar will be applied to scores: `scores *= scale`.
    3. Use scores to calculate a distribution with shape
      `[batch_size, Tq, Tk]`: `distribution = tf.nn.softmax(scores)`.
    4. Use `distribution` to create a linear combination of `Value` with
      shape `batch_size, Tq, dim]`:
      `return tf.matmul(distribution, V)`.

    Args:
      inputs: List of the following tensors:
        * query: Query `Tensor` of shape `[batch_size, Tq, dim]`.
        * value: Value `Tensor` of shape `[batch_size, Tv, dim]`.
        * key: Optional key `Tensor` of shape `[batch_size, Tv, dim]`. If not
          given, will use `value` for both `key` and `value`, which is the
          most common case.
      mask: List of the following tensors:
        * query_mask: A boolean mask `Tensor` of shape `[batch_size, Tq]`.
          If given, the output will be zero at the positions where
          `mask==False`.
        * value_mask: A boolean mask `Tensor` of shape `[batch_size, Tv]`.
          If given, will apply the mask such that values at positions where
          `mask==False` do not contribute to the result.
      key_mask: A boolean mask `Tensor` of shape `[batch_size, Tk]`.
        If given, will apply the mask such that Value at positions where
        `mask==False` do not contribute to the result.
      causal: Boolean. Set to `True` for decoder self-attention. Adds a mask such
        that position `i` cannot attend to positions `j > i`. This prevents the
        flow of information from the future towards the past.
      use_scale: Boolean. If `True`, will create a scalar variable to scale the
        attention scores.
      softmax_dtype: Do softmax in FP32 data type to improve accuracy in BF16 training.

  Output shape:
    Outputs of shape `[batch_size, Tq, dim]`.
  '''
  layer = Attention(causal=causal, use_scale=use_scale, name=name, _scope=name, _reuse=reuse)
  return layer.apply(inputs, mask)


@tf_export(v1=['layers.self_attention'])
def self_attention(query,
                   key=None,
                   value=None,
                   dim=None,
                   query_mask=None,
                   key_mask=None,
                   causal=False,
                   use_scale=False,
                   name='self_attention'):
  '''Self attention layer

    Inputs are `query` tensor of shape `[batch_size, Tq, ?]`, `value` tensor of
    shape `[batch_size, Tk, ?]` and `key` tensor of shape
    `[batch_size, Tk, ?]`. The calculation follows the steps:

    1. Calculute the linear projections `Q` of shape `[batch_size, Tq, dim]` for `query`, 
      `K` and `V` of shape `[batch_size, Tk, dim]` for `key` and `value`
    2. Use linear projections to calculate scores with shape `[batch_size, Tq, Tv]` 
      as a `query`-`key` dot product: `scores = tf.matmul(Q, K, transpose_b=True)`.
    2. Use scores to calculate a distribution with shape
      `[batch_size, Tq, Tk]`: `distribution = tf.nn.softmax(scores)`.
    3. Use `distribution` to create a linear combination of `value` with
      shape `batch_size, Tq, dim]`:
      `return tf.matmul(distribution, V)`.

    Args:
      query: Query `Tensor` of shape `[batch_size, Tq, ?]`.
      key: Optional key `Tensor` of shape `[batch_size, Tk, ?]`. If not given,
        will use `query` for `query`, `key` and `value`.
      value: Optional value `Tensor` of shape `[batch_size, Tk, ?]`. If not
        given, will use `key` for both `key` and `value`, which is the
        most common case.
      dim: Dimension for attention size used to create a linear projections of shape 
        `[batch_size, Tq, ?]` for Q,K,V.
      query_mask: A boolean mask `Tensor` of shape `[batch_size, Tq]`.
        If given, the output will be zero at the positions where
        `mask==False`.
      key_mask: A boolean mask `Tensor` of shape `[batch_size, Tk]`.
        If given, will apply the mask such that value at positions where
        `mask==False` do not contribute to the result.
      causal: Boolean. Set to `True` for decoder self-attention. Adds a mask such
        that position `i` cannot attend to positions `j > i`. This prevents the
        flow of information from the future towards the past.
      use_scale: Boolean. If `True`, will create a scalar variable to scale the
        attention scores.
      softmax_dtype: Do softmax in FP32 data type to improve accuracy in BF16 training.

  Output shape:

  Self attention outputs of shape `[batch_size, Tq, dim]`.
  '''
  if key is None and value is not None:
    raise ValueError("key must be provided when value is not empty.")
  if dim is None:  # is int
    dim = array_ops.shape(query)[-1]
  elif type(dim) is not int:
    raise ValueError("Dim must be a integer!")
  elif dim <= 0:
    raise ValueError("Dim be greater than 0.")

  _query = query
  _key = key if key is not None else _query
  _value = value if value is not None else _key

  with ops.name_scope(name, 'self_attention') as scope_name:
    # Q shape = [batch_size, Tq, dim]
    # KV shape = [batch_size, Tk, dim]
    # TODO: check are Q,K,V the same. by tf.equal()
    if key is None:
      Query, Key, Value = _fused_QKV_dense(_query, dim)
    elif value is None:
      Query = core_layers.dense(_query, dim, activation=nn.relu, name='Query')
      Key, Value = _fused_KV_dense(_key, dim)
    else:
      Query = core_layers.dense(_query, dim, activation=nn.relu, name='Query')
      Key = core_layers.dense(_key, dim, activation=nn.relu, name='Key')
      Value = core_layers.dense(_value, dim, activation=nn.relu, name='Value')

    result = attention(inputs=[Query, Value, Key],
                       mask=[query_mask, key_mask],
                       use_scale=use_scale,
                       causal=causal)
  return result


@tf_export(v1=['layers.multihead_attention'])
def multihead_attention(query,
                        key=None,
                        value=None,
                        head_count=None,
                        dim=None,
                        query_mask=None,
                        key_mask=None,
                        causal=False,
                        use_scale=False,
                        name='multihead_attention'):
  '''Multihead attention layer

    Args:
      query: Query `Tensor` of shape `[batch_size, Tq, ?]`.
      key: Optional key `Tensor` of shape `[batch_size, Tk, ?]`. If not given,
        will use `query` for `query`, `key` and `value`.
      value: Optional value `Tensor` of shape `[batch_size, Tk, ?]`. If not
        given, will use `key` for both `key` and `value`, which is the
        most common case.
      head_count: Integer nums of multi head counts.
      dim: Dimension for attention size used to create a linear projections of shape 
        `[batch_size, Tq, ?]` for Q,K,V.
      query_mask: A boolean mask `Tensor` of shape `[batch_size, Tq]`.
        If given, the output will be zero at the positions where
        `mask==False`.
      key_mask: A boolean mask `Tensor` of shape `[batch_size, Tk]`.
        If given, will apply the mask such that value at positions where
        `mask==False` do not contribute to the result.
      causal: Boolean. Set to `True` for decoder self-attention. Adds a mask such
        that position `i` cannot attend to positions `j > i`. This prevents the
        flow of information from the future towards the past.
      use_scale: Boolean. If `True`, will create a scalar variable to scale the
        attention scores.
      softmax_dtype: Do softmax in FP32 data type to improve accuracy in BF16 training.

  Output shape:

  Multihead attention outputs of shape `[batch_size, Tq, dim]`.
  '''
  if key is None and value is not None:
    raise ValueError("key must be provided when value is not empty.")
  if dim is None:
    dim = array_ops.shape(query)[-1]
  elif type(dim) is not int:
    raise ValueError("Dim must be a integer!")
  elif dim <= 0:
    raise ValueError("Dim be greater than 0.")

  if head_count is None:
    raise ValueError("Head count must be provided.")
  elif type(head_count) is not int:
    raise ValueError("Head count must be a integer!")
  elif head_count < 1:
    raise ValueError("Head count must be greater than 2.")
  elif head_count == 1:
    raise ValueError("Head count is equel to 1, please use layers.self_attention.")

  if dim % head_count != 0:
    raise ValueError("Dim must be divisible by head count.")

  _query = query
  _key = key if key is not None else _query
  _value = value if value is not None else _key

  with ops.name_scope(name, 'multihead_attention') as scope_name:
    # Q shape = [batch_size, Tq, dim]
    # KV shape = [batch_size, Tk, dim]
    # TODO: check are Q,K,V the same. by tf.equal()
    if key is None:
      Query, Key, Value = _fused_QKV_dense(_query, dim)
    elif value is None:
      Query = core_layers.dense(_query, dim, activation=nn.relu, name='Query')
      Key, Value = _fused_KV_dense(_key, dim)
    else:
      Query = core_layers.dense(_query, dim, activation=nn.relu, name='Query')
      Key = core_layers.dense(_key, dim, activation=nn.relu, name='Key')
      Value = core_layers.dense(_value, dim, activation=nn.relu, name='Value')

      # Q shape = [batch_size * head_count, Tq, dim / head_count]
      # KV shape = [batch_size * head_count, Tk, dim / head_count]
    Query = _fused_split_concat(Query, head_count, -1, 0)
    Key = _fused_split_concat(Key, head_count, -1, 0)
    Value = _fused_split_concat(Value, head_count, -1, 0)

    if key_mask is not None:
      # key_mask shape = [batch_size * head_count, Tk]
      key_mask = array_ops.tile(key_mask, [head_count, 1])
    if query_mask is not None:
      # key_mask shape = [batch_size * head_count, Tq]
      query_mask = array_ops.tile(query_mask, [head_count, 1])

    # att_res_net shape = [batch_size * head_count, Tq, dim / head_count]

    att_res_net = attention(inputs=[Query, Value, Key],
                            mask=[query_mask, key_mask],
                            use_scale=use_scale,
                            causal=causal)

    result = _fused_split_concat(att_res_net, head_count, 0, 2)
    # result shape = [batch_size, Tq, dim]
    result = core_layers.dense(result, dim, activation=nn.relu)
  return result
