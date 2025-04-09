import tensorflow as tf
import numpy as np
import os
import collections
from . import test_base
from .test_base import ShapeDesc

from .. import flash_attention


class SyncModeNoneFront(object):
  _SYNC_MODE = 'none_front'

  def _compute_locations(self, Q_shape, K_shape):
    #max_height = tf.maximum(Q_shape[-2], K_shape[-2])
    max_width = tf.maximum(Q_shape[-1], K_shape[-1])

    Q_y, Q_x = tf.meshgrid(tf.range(Q_shape[-2], dtype=tf.int32), tf.range(Q_shape[-1], dtype=tf.int32), indexing='ij')

    Q_l = Q_y * max_width + Q_x

    K_y, K_x = tf.meshgrid(tf.range(K_shape[-2], dtype=tf.int32), tf.range(K_shape[-1], dtype=tf.int32), indexing='ij')

    K_l = K_y * max_width + K_x

    return (tf.stack((Q_y, Q_x), axis=-1), tf.stack((K_y, K_x), axis=-1)), (Q_l, K_l)


class SyncModeScaleFront(object):
  _SYNC_MODE = 'scale_front'

  def _compute_locations(self, Q_shape, K_shape):
    max_height = tf.maximum(Q_shape[-2], K_shape[-2])
    max_width = tf.maximum(Q_shape[-1], K_shape[-1])

    Q_y, Q_x = tf.meshgrid(tf.range(Q_shape[-2], dtype=tf.int32), tf.range(Q_shape[-1], dtype=tf.int32), indexing='ij')
    y_step = max_height // Q_shape[-2]
    x_step = max_width // Q_shape[-1]

    Q_y = Q_y * y_step
    Q_x = Q_x * x_step
    Q_l = Q_y * max_width + Q_x

    K_y, K_x = tf.meshgrid(tf.range(K_shape[-2], dtype=tf.int32), tf.range(K_shape[-1], dtype=tf.int32), indexing='ij')
    y_step = max_height // K_shape[-2]
    x_step = max_width // K_shape[-1]

    K_y = K_y * y_step
    K_x = K_x * x_step
    K_l = K_y * max_width + K_x

    return (tf.stack((Q_y, Q_x), axis=-1), tf.stack((K_y, K_x), axis=-1)), (Q_l, K_l)


class SyncModeScaleEnd(object):
  _SYNC_MODE = 'scale_end'

  def _compute_locations(self, Q_shape, K_shape):
    max_height = tf.maximum(Q_shape[-2], K_shape[-2])
    max_width = tf.maximum(Q_shape[-1], K_shape[-1])

    Q_y, Q_x = tf.meshgrid(tf.range(Q_shape[-2], dtype=tf.int32), tf.range(Q_shape[-1], dtype=tf.int32), indexing='ij')
    y_step = max_height // Q_shape[-2]
    x_step = max_width // Q_shape[-1]

    Q_y = (Q_y+1) * y_step - 1
    Q_x = (Q_x+1) * x_step - 1
    Q_l = Q_y * max_width + Q_x

    K_y, K_x = tf.meshgrid(tf.range(K_shape[-2], dtype=tf.int32), tf.range(K_shape[-1], dtype=tf.int32), indexing='ij')
    y_step = max_height // K_shape[-2]
    x_step = max_width // K_shape[-1]

    K_y = (K_y+1) * y_step - 1
    K_x = (K_x+1) * x_step - 1
    K_l = K_y * max_width + K_x

    return (tf.stack((Q_y, Q_x), axis=-1), tf.stack((K_y, K_x), axis=-1)), (Q_l, K_l)


class Test(object):
  class Base(test_base.Test.Base):

    _SEQUENCE_DIMS = 2
    _SHAPE_DESC_TABLE = {
      tf.float16: ShapeDesc(tf.constant([1, 8, 8, 16, 16], dtype=tf.int32),
                            tf.constant([1, 8, 32, 64, 64], dtype=tf.int32)),

      tf.float32: ShapeDesc(tf.constant([1, 8, 8, 16, 16], dtype=tf.int32),
                            tf.constant([1, 8, 32, 32, 64], dtype=tf.int32)),

      tf.float64: ShapeDesc(tf.constant([1, 8, 8, 16, 16], dtype=tf.int32),
                            tf.constant([1, 8, 32, 32, 32], dtype=tf.int32))
    }


    def _vanilla_attention(self, Q, K, V, mask):
      Q_shape = tf.shape(Q)
      K_shape = tf.shape(K)


      logit = tf.einsum('bhcyx,bhcts->bhyxts', Q, K) / tf.sqrt(tf.cast(Q_shape[-self._SEQUENCE_DIMS-1], dtype=Q.dtype))
      logit = tf.where(mask, logit, logit.dtype.min)
      logit_shape = tf.shape(logit)
      logit = tf.reshape(logit, tf.concat((logit_shape[:-2], [-1]), axis=-1))
      p = tf.nn.softmax(logit, axis=-1)
      p = tf.reshape(p, logit_shape)
      p = tf.where(mask, p, tf.constant(0.0, dtype=p.dtype))
      return tf.einsum('bhyxts,bhcts->bhcyx', p, V), p

  class FullAttentionBase(test_base.VanillaFullnessPolicy, Base):
    def _flash_attention(self, Q, K, V):
      return flash_attention.full_2d(Q, K, V, sync_mode=self._SYNC_MODE, returning_l_m=True)

  class CausalAttentionBase(test_base.VanillaCausalityPolicy, Base):
    def _flash_attention(self, Q, K, V):
      return flash_attention.causal_2d(Q, K, V, sync_mode=self._SYNC_MODE, returning_l_m=True)

  class LocalAttentionBase(test_base.VanillaLocalPolicy, Base):

    def _flash_attention(self, Q, K, V):
      return flash_attention.local_2d(Q, K, V, window_size=self._window_size, log2_stride_size=self._log2_stride_size, is_causal=self._IS_CAUSAL, sync_mode=self._SYNC_MODE, returning_l_m=True)

  class LocalStrideAttentionBase(LocalAttentionBase):
    _STRIDE_SIZE_ABOVE_1 = True

  class LocalAndCausalAttentionBase(LocalAttentionBase):
    _IS_CAUSAL = True

  class LocalStrideAndCausalAttentionBase(LocalAttentionBase):
    _STRIDE_SIZE_ABOVE_1 = True
    _IS_CAUSAL = True


class TestGroup(test_base.TestGroup):
  '''
  TestGroup contains all tests/benchmarks.

  To list tests in the TestGroup:
  python -m flash_attention.tests.test_2d TestGroup.list

  To launch all tests at once:
  python -m flash_attention.tests.test_2d TestGroup.verify

  To launch one specific test, for example the test named "FullAttentionSyncModeNoneFront":
  TESTCASE=FullAttentionSyncModeNoneFront python -m flash_attention.tests.test_2d TestGroup.verify

  benchmarks can also be launched in a similar fashion,
  by replacing "TestGroup.verify" with "TestGroup.benchmark"
  '''

  def __init__(self, *args, **kwargs):
    sync_classes = test_base.SyncClasses(
      SyncModeNoneFront,
      SyncModeScaleFront,
      SyncModeScaleEnd
    )

    attention_classes = test_base.AttentionClasses(
      Test.FullAttentionBase,
      Test.CausalAttentionBase,
      Test.LocalAttentionBase,
      Test.LocalStrideAttentionBase,
      Test.LocalAndCausalAttentionBase,
      Test.LocalStrideAndCausalAttentionBase,
    )

    super().__init__(sync_classes, attention_classes,
                      *args, **kwargs)

if __name__ == "__main__":
  import time

  test_base.random_seed = int(time.time())
  print(f'random seed = {test_base.random_seed}')
  tf.test.main()

