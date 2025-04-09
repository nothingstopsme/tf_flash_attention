'''
This module provides interfaces to the kernel implementing the forward/backward process
of flash_attention. Note that in this implementation, masking patterns are defined by rules
as opposed to mask tensors, and currently there are 3 types of masking available:

1. full: No masking
2. causal: The pattern which only allows to look at the current position or behind
3. local: The configurable pattern with 3 parameters: window size, stride size
(has to be a power of 2 >= 1), and whether it is causal or not.

Given a pair of entries, masking rules are checked based on
1. The causality relationship of the two entries, which is determined by their indices in
their respective sequences flattened in row-major order.
2. Coordinate differences. e.g. Local masking requires magnitudes and signs of
such differences meet window constraints

To further add flexibility to ordering/indexing and resultant masking behaviours, there are also 3
synchronisation modes available to dictate how entries in two sequence are aligned when the sizes
of their sequence dimensions are different:

1. none_front: Coordinates not scaled (step size = 1) and aligned at the front of each step along
respective dimensions.
2. scale_front: Coordinates scaled per dimension ratios and aligned at the front of each step
along respective dimensions.
3. scale_end: Coordinates scaled per dimension ratios and aligned at the end of each step along
respective dimensions.

Below are some examples:

When two 1d sequences synchronised in none_front mode are
aligned as follows (corresponding entries have the same indices)
A: [0, 1, 2, 3, 4, 5]
B: [0, 1, 2]

then in scale_front mode their alignment becomes
A: [0, 1, 2, 3, 4, 5]
B: [0, 2, 4]

while in scale_end mode it is
A: [0, 1, 2, 3, 4, 5]
B: [1, 3, 5]

On the other hand, when two 2d sequences synchronised in none_front mode are
aligned as follows (corresponding entries have the same indices)
A: [[0, 1, 2, 3]
    [4, 5, 6, 7]
    [8, 9, 10, 11]
    [12, 13, 14, 15]]

B: [[0, 1],
    [4, 5]]

then in scale_front mode their alignment becomes
A: [[0, 1, 2, 3]
    [4, 5, 6, 7]
    [8, 9, 10, 11]
    [12, 13, 14, 15]]

B: [[0, 2],
    [8, 10]]

while in scale_end mode it is
A: [[0, 1, 2, 3]
    [4, 5, 6, 7]
    [8, 9, 10, 11]
    [12, 13, 14, 15]]

B: [[5, 7],
    [13, 15]]
'''

import tensorflow as tf
import os, re
from tensorflow.python.framework import ops


_script_dir = os.path.abspath(os.path.dirname(__file__))
_fa_kernel = tf.load_op_library(os.path.join(_script_dir, 'kernel/flash_attention.so'))

def full_1d(Q, K, V, sync_mode='none_front', returning_l_m=False):
  '''Conducting full attention (no masking) on 1d sequences

  Args:
    Q: The query tensor of a shape = batch_shape + (query/key channel dimension, query sequence dimension).
    Note that batch_shape can include the dimension of attention heads, and be of any dimensionality.

    K: The key tensor of a shape = batch_shape + (query/key channel dimension, key/value sequence dimension).
    The batch_shape must match the one of Q.

    V: The value tensor of a shape = batch_shape + (value channel dimension, key/value sequence dimension).
    The batch_shape must match the one of Q and K.

    sync_mode: how entries in Q and K are aligned when the sizes of their sequence dimensions are different.
    Note that it does not really matter which sync_mode is used as this only affects masking, and masking
    is not applied in full attention. Default to 'none_front'.

    returning_l_m: Whether or not to return normalisers (l) and maximum logits (m) of attention distributions.
    This is primarily for the purpose of testing and is default to False.

  Returns:
    (O, l, m) if returning_l_m is True; otherwise O.

    O: The attention result tensor of a shape
    = batch_shape + (value channel dimension, query sequence dimension).

    l: A tensor of a shape
    = batch_shape + (query sequence dimension,), containing normalisers of attention distributions.

    m: A tensor of a shape
    = batch_shape + (query sequence dimension,), containing maximum logits of attention distributions.
  '''

  if Q.dtype == tf.float16:
    attend_fn = _fa_kernel.full_attention_forward1d_float16
  else:
    attend_fn = _fa_kernel.full_attention_forward1d

  results = attend_fn(Q, K, V, sync_mode=sync_mode)
  return results if returning_l_m else results[0]


def causal_1d(Q, K, V, sync_mode, returning_l_m=False):
  '''Conducting causal attention on 1d sequences

  Args:
    Q: The query tensor of a shape = batch_shape + (query/key channel dimension, query sequence dimension).
    Note that batch_shape can include the dimension of attention heads, and be of any dimensionality.

    K: The key tensor of a shape = batch_shape + (query/key channel dimension, key/value sequence dimension).
    The batch_shape must match the one of Q.

    V: The value tensor of a shape = batch_shape + (value channel dimension, key/value sequence dimension).
    The batch_shape must match the one of Q and K.

    sync_mode: how entries in Q and K are aligned when the sizes of their sequence dimensions are different.
    Please check the module description above for more details.

    returning_l_m: Whether or not to return normalisers (l) and maximum logits (m) of attention distributions.
    This is primarily for the purpose of testing and is default to False.

  Returns:
    (O, l, m) if returning_l_m is True; otherwise O.

    O: The attention result tensor of a shape
    = batch_shape + (value channel dimension, query sequence dimension).

    l: A tensor of a shape
    = batch_shape + (query sequence dimension,), containing normalisers of attention distributions.

    m: A tensor of a shape
    = batch_shape + (query sequence dimension,), containing maximum logits of attention distributions.
  '''

  if Q.dtype == tf.float16:
    attend_fn = _fa_kernel.causal_attention_forward1d_float16
  else:
    attend_fn = _fa_kernel.causal_attention_forward1d

  results = attend_fn(Q, K, V, sync_mode=sync_mode)
  return results if returning_l_m else results[0]


def local_1d(Q, K, V, window_size, log2_stride_size, is_causal, sync_mode, returning_l_m=False):
  '''Conducting local attention on 1d sequences

  Args:
    Q: The query tensor of a shape = batch_shape + (query/key channel dimension, query sequence dimension).
    Note that batch_shape can include the dimension of attention heads, and be of any dimensionality.

    K: The key tensor of a shape = batch_shape + (query/key channel dimension, key/value sequence dimension).
    The batch_shape must match the one of Q.

    V: The value tensor of a shape = batch_shape + (value channel dimension, key/value sequence dimension).
    The batch_shape must match the one of Q and K.

    window_size: (Approximately) half the size of attention windows. In 1d cases, attention windows
    are 1-dimensional with their length equal to 2 * window_size - 1, centred at each query.
    Note that when is_causal is set to True, entries in windows not meeting the causality constraint will
    also be masked out, resulting in (roughly) half the areas of full windows being used only.

    log2_stride_size: The exponent of a power of 2, describing the stride size between any two neighbouring
    entries in an attention window. This value should be >= 0 and < 31 (outcomes of calculation with it need to be
    representable by int32), and when it is not 0, the corresponding masking pattern becomes discontinuous.

    is_causal: A boolean value indicating whether to maintain the causality property, i.e. whether attention
    can only looks at the current position or behind, but not ahead, with respect to the causality relationship
    described by given sequences. Note that this value will change how window_size is interpreted;
    please check the section of window_size for more details.

    sync_mode: how entries in Q and K are aligned when the sizes of their sequence dimensions are different.
    Please check the module description above for more details.

    returning_l_m: Whether or not to return normalisers (l) and maximum logits (m) of attention distributions.
    This is primarily for the purpose of testing and is default to False

  Returns:
    (O, l, m) if returning_l_m is True; otherwise O.

    O: The attention result tensor of a shape
    = batch_shape + (value channel dimension, query sequence dimension).

    l: A tensor of a shape
    = batch_shape + (query sequence dimension,), containing normalisers of attention distributions.

    m: A tensor of a shape
    = batch_shape + (query sequence dimension,), containing maximum logits of attention distributions.
  '''

  if Q.dtype == tf.float16:
    attend_fn = _fa_kernel.local_attention_forward1d_float16
  else:
    attend_fn = _fa_kernel.local_attention_forward1d

  results = attend_fn(Q, K, V, window_size=window_size, log2_stride_size=log2_stride_size,
                      is_causal=is_causal, sync_mode=sync_mode)
  return results if returning_l_m else results[0]


def full_2d(Q, K, V, sync_mode='none_front', returning_l_m=False):
  '''Conducting full attention (no masking) on 2d sequences

  Args:
    Q: The query tensor of a shape
    = batch_shape + (query/key channel dimension, query sequence dimension0, query sequence dimension1).
    Note that batch_shape can include the dimension of attention heads, and be of any dimensionality.

    K: The key tensor of a shape
    = batch_shape + (query/key channel dimension, key/value sequence dimension0, key/value sequence dimension1).
    The batch_shape must match the one of Q.

    V: The value tensor of a shape
    = batch_shape + (value channel dimension, key/value sequence dimension0, key/value sequence dimension1).
    The batch_shape must match the one of Q and K.

    sync_mode: how entries in Q and K are aligned when the sizes of their sequence dimensions are different.
    Note that it does not really matter which sync_mode is used as this only affects masking, and masking
    is not applied in full attention. Default to 'none_front'

    returning_l_m: Whether or not to return normalisers (l) and maximum logits (m) of attention distributions.
    This is primarily for the purpose of testing and is default to False

  Returns:
    (O, l, m) if returning_l_m is True; otherwise O.

    O: The attention result tensor of a shape
    = batch_shape + (value channel dimension, query sequence dimension0, query sequence dimension1).

    l: A tensor of a shape
    = batch_shape + (query sequence dimension0, query sequence dimension1), containing normalisers of
    attention distributions.

    m: A tensor of a shape
    = batch_shape + (query sequence dimension0, query sequence dimension1), containing maximum logits of
    attention distributions.
  '''

  if Q.dtype == tf.float16:
    attend_fn = _fa_kernel.full_attention_forward2d_float16
  else:
    attend_fn = _fa_kernel.full_attention_forward2d

  results = attend_fn(Q, K, V, sync_mode=sync_mode)
  return results if returning_l_m else results[0]


def causal_2d(Q, K, V, sync_mode, returning_l_m=False):
  '''Conducting causal attention on 2d sequences

  Args:
    Q: The query tensor of a shape
    = batch_shape + (query/key channel dimension, query sequence dimension0, query sequence dimension1).
    Note that batch_shape can include the dimension of attention heads, and be of any dimensionality.

    K: The key tensor of a shape
    = batch_shape + (query/key channel dimension, key/value sequence dimension0, key/value sequence dimension1).
    The batch_shape must match the one of Q.

    V: The value tensor of a shape
    = batch_shape + (value channel dimension, key/value sequence dimension0, key/value sequence dimension1).
    The batch_shape must match the one of Q and K.

    sync_mode: how entries in Q and K are aligned when the sizes of their sequence dimensions are different.
    Please check the module description above for more details.

    returning_l_m: Whether or not to return normalisers (l) and maximum logits (m) of attention distributions.
    This is primarily for the purpose of testing and is default to False

  Returns:
    (O, l, m) if returning_l_m is True; otherwise O.

    O: The attention result tensor of a shape
    = batch_shape + (value channel dimension, query sequence dimension0, query sequence dimension1).

    l: A tensor of a shape
    = batch_shape + (query sequence dimension0, query sequence dimension1), containing normalisers of
    attention distributions.

    m: A tensor of a shape
    = batch_shape + (query sequence dimension0, query sequence dimension1), containing maximum logits of
    attention distributions.
  '''

  if Q.dtype == tf.float16:
    attend_fn = _fa_kernel.causal_attention_forward2d_float16
  else:
    attend_fn = _fa_kernel.causal_attention_forward2d

  results = attend_fn(Q, K, V, sync_mode=sync_mode)
  return results if returning_l_m else results[0]


def local_2d(Q, K, V, window_size, log2_stride_size, is_causal, sync_mode, returning_l_m=False):
  '''Conducting local attention on 2d sequences

  Args:
    Q: The query tensor of a shape
    = batch_shape + (query/key channel dimension, query sequence dimension0, query sequence dimension1).
    Note that batch_shape can include the dimension of attention heads, and be of any dimensionality.

    K: The key tensor of a shape
    = batch_shape + (query/key channel dimension, key/value sequence dimension0, key/value sequence dimension1).
    The batch_shape must match the one of Q.

    V: The value tensor of a shape
    = batch_shape + (value channel dimension, key/value sequence dimension0, key/value sequence dimension1).
    The batch_shape must match the one of Q and K.

    window_size: (Approximately) half the size of attention windows. In 2d cases, attention windows
    become 2-dimensional with their height/width equal to 2 * window_size - 1, centred at each query.
    Note that when is_causal is set to True, entries in windows not meeting the causality constraint will
    also be masked out, resulting in (roughly) half the areas of full windows being used only.

    log2_stride_size: The exponent of a power of 2, describing the stride size between any two neighbouring
    entries in an attention window. This value should be >= 0 and < 31 (outcomes of calculation with it need to be
    representable by int32), and when it is not 0, the corresponding masking pattern becomes discontinuous.

    is_causal: A boolean value indicating whether to maintain the causality property, i.e. whether attention
    can only looks at the current position or behind, but not ahead, with respect to the causality relationship
    described by given sequences. Note that this value will change how window_size is interpreted;
    please check the section of window_size for more details.

    sync_mode: how entries in Q and K are aligned when the sizes of their sequence dimensions are different.
    Please check the module description above for more details.

    returning_l_m: Whether or not to return normalisers (l) and maximum logits (m) of attention distributions.
    This is primarily for the purpose of testing and is default to False

  Returns:
    (O, l, m) if returning_l_m is True; otherwise O.

    O: The attention result tensor of a shape
    = batch_shape + (value channel dimension, query sequence dimension0, query sequence dimension1).

    l: A tensor of a shape
    = batch_shape + (query sequence dimension0, query sequence dimension1), containing normalisers of
    attention distributions.

    m: A tensor of a shape
    = batch_shape + (query sequence dimension0, query sequence dimension1), containing maximum logits of
    attention distributions.

  '''

  if Q.dtype == tf.float16:
    attend_fn = _fa_kernel.local_attention_forward2d_float16
  else:
    attend_fn = _fa_kernel.local_attention_forward2d

  results = attend_fn(Q, K, V, window_size=window_size, log2_stride_size=log2_stride_size, is_causal=is_causal, sync_mode=sync_mode)
  return results if returning_l_m else results[0]

_OP_RE = re.compile(r'.+(?P<ndim>\dd)(?P<f16>Float16)?$')

def _compute_gradients(grad_fn, op, *grads, **extra_kwargs):
  Q = op.inputs[0]
  K = op.inputs[1]
  V = op.inputs[2]
  O = op.outputs[0]
  l = op.outputs[1]
  m = op.outputs[2]

  # Only gradients propagated from the attention output O are considered,
  # as l and m are simply caches for facilating gradient computation
  dO = grads[0]

  #extra_kwargs['sequence_dims'] = op.get_attr('sequence_dims')
  #extra_kwargs['alibi_on'] = op.get_attr('alibi_on')
  extra_kwargs['sync_mode'] = op.get_attr('sync_mode')

  return grad_fn(Q, K, V, O, l, m, dO, **extra_kwargs)

@ops.RegisterGradient("FullAttentionForward1dFloat16")
@ops.RegisterGradient("FullAttentionForward1d")
@ops.RegisterGradient("FullAttentionForward2dFloat16")
@ops.RegisterGradient("FullAttentionForward2d")
def _full_gradients(op, *grads):
  grad_fn = None
  match = _OP_RE.match(op.type)
  if match:
    match = match.groupdict()
    if match['ndim'] == '1d':
      if match['f16']:
        grad_fn = _fa_kernel.full_attention_backward1d_float16
      else:
        grad_fn = _fa_kernel.full_attention_backward1d
    elif match['ndim'] == '2d':
      if match['f16']:
        grad_fn = _fa_kernel.full_attention_backward2d_float16
      else:
        grad_fn = _fa_kernel.full_attention_backward2d

  if grad_fn is None:
    raise ValueError(f'Unsupported op "{op.type}"')

  return _compute_gradients(grad_fn, op, *grads)


@ops.RegisterGradient("CausalAttentionForward1dFloat16")
@ops.RegisterGradient("CausalAttentionForward1d")
@ops.RegisterGradient("CausalAttentionForward2dFloat16")
@ops.RegisterGradient("CausalAttentionForward2d")
def _causal_gradients(op, *grads):
  grad_fn = None
  match = _OP_RE.match(op.type)
  if match:
    match = match.groupdict()
    if match['ndim'] == '1d':
      if match['f16']:
        grad_fn = _fa_kernel.causal_attention_backward1d_float16
      else:
        grad_fn = _fa_kernel.causal_attention_backward1d
    elif match['ndim'] == '2d':
      if match['f16']:
        grad_fn = _fa_kernel.causal_attention_backward2d_float16
      else:
        grad_fn = _fa_kernel.causal_attention_backward2d

  if grad_fn is None:
    raise ValueError(f'Unsupported op "{op.type}"')

  return _compute_gradients(grad_fn, op, *grads)



@ops.RegisterGradient("LocalAttentionForward1dFloat16")
@ops.RegisterGradient("LocalAttentionForward1d")
@ops.RegisterGradient("LocalAttentionForward2dFloat16")
@ops.RegisterGradient("LocalAttentionForward2d")
def _local_gradients(op, *grads):
  grad_fn = None
  match = _OP_RE.match(op.type)
  if match:
    match = match.groupdict()
    if match['ndim'] == '1d':
      if match['f16']:
        grad_fn = _fa_kernel.local_attention_backward1d_float16
      else:
        grad_fn = _fa_kernel.local_attention_backward1d
    else:
      if match['f16']:
        grad_fn = _fa_kernel.local_attention_backward2d_float16
      else:
        grad_fn = _fa_kernel.local_attention_backward2d

  if grad_fn is None:
    raise ValueError(f'Unsupported op "{op.type}"')

  return _compute_gradients(grad_fn, op, *grads,
                            window_size=op.get_attr('window_size'),
                            log2_stride_size=op.get_attr('log2_stride_size'),
                            is_causal=op.get_attr('is_causal'))



def _estimate_forward_flops(flops_fn, graph, node, **extra_kwargs):

  def get_tensor_from_gragh(name):
    nonlocal graph

    if ":" not in name:
      canonical_name = name + ":0"
    else:
      canonical_name = name

    return graph.get_tensor_by_name(canonical_name)

  Q = get_tensor_from_gragh(node.input[0])
  K = get_tensor_from_gragh(node.input[1])
  V = get_tensor_from_gragh(node.input[2])

  #extra_kwargs['sequence_dims'] = node.attr['sequence_dims'].i
  #extra_kwargs['alibi_on'] = node.attr['alibi_on'].b
  extra_kwargs['sync_mode'] = node.attr['sync_mode'].s.decode()

  flops = flops_fn(Q.shape, K.shape, V.shape, dtype=Q.dtype, **extra_kwargs)
  return ops.OpStats("flops", flops)


@ops.RegisterStatistics("FullAttentionForward1dFloat16", "flops")
@ops.RegisterStatistics("FullAttentionForward1d", "flops")
@ops.RegisterStatistics("FullAttentionForward2dFloat16", "flops")
@ops.RegisterStatistics("FullAttentionForward2d", "flops")
def _estimate_full_attention_forward_flops(graph, node):
  estimate_fn = None
  match = _OP_RE.match(node.op)
  if match:
    match = match.groupdict()
    if match['ndim'] == '1d':
      estimate_fn = _fa_kernel.estimate_full_attention_forward1d_flops
    elif match['ndim'] == '2d':
      estimate_fn = _fa_kernel.estimate_full_attention_forward2d_flops

  if estimate_fn is None:
    raise ValueError(f'Unsupported op "{node.op}"')

  return _estimate_forward_flops(estimate_fn,
                                  graph, node)


@ops.RegisterStatistics("CausalAttentionForward1dFloat16", "flops")
@ops.RegisterStatistics("CausalAttentionForward1d", "flops")
@ops.RegisterStatistics("CausalAttentionForward2dFloat16", "flops")
@ops.RegisterStatistics("CausalAttentionForward2d", "flops")
def _estimate_causal_attention_forward_flops(graph, node):
  estimate_fn = None
  match = _OP_RE.match(node.op)
  if match:
    match = match.groupdict()
    if match['ndim'] == '1d':
      estimate_fn = _fa_kernel.estimate_causal_attention_forward1d_flops
    elif match['ndim'] == '2d':
      estimate_fn = _fa_kernel.estimate_causal_attention_forward2d_flops

  if estimate_fn is None:
    raise ValueError(f'Unsupported op "{node.op}"')

  return _estimate_forward_flops(estimate_fn,
                                  graph, node)


@ops.RegisterStatistics("LocalAttentionForward1dFloat16", "flops")
@ops.RegisterStatistics("LocalAttentionForward1d", "flops")
@ops.RegisterStatistics("LocalAttentionForward2dFloat16", "flops")
@ops.RegisterStatistics("LocalAttentionForward2d", "flops")
def _estimate_local_attention_forward_flops(graph, node):
  estimate_fn = None
  match = _OP_RE.match(node.op)
  if match:
    match = match.groupdict()
    if match['ndim'] == '1d':
      estimate_fn = _fa_kernel.estimate_local_attention_forward1d_flops
    elif match['ndim'] == '2d':
      estimate_fn = _fa_kernel.estimate_local_attention_forward2d_flops

  if estimate_fn is None:
    raise ValueError(f'Unsupported op "{node.op}"')

  return _estimate_forward_flops(estimate_fn,
                                  graph, node,
                                  window_size=node.attr['window_size'].i,
                                  log2_stride_size=node.attr['log2_stride_size'].i,
                                  is_causal=node.attr['is_causal'].b)



