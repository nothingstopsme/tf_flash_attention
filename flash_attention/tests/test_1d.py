import tensorflow as tf
import numpy as np
import collections
from . import test_base
from .test_base import ShapeDesc

from .. import flash_attention

class SyncModeNoneFront(object):
  _SYNC_MODE = 'none_front'

  def _compute_locations(self, Q_shape, K_shape):
    Q_l = tf.range(Q_shape[-1], dtype=tf.int32)
    K_l = tf.range(K_shape[-1], dtype=tf.int32)

    return (Q_l[..., tf.newaxis], K_l[..., tf.newaxis]), (Q_l, K_l)



class SyncModeScaleFront(object):
  _SYNC_MODE = 'scale_front'

  def _compute_locations(self, Q_shape, K_shape):
    max_length = tf.maximum(Q_shape[-1], K_shape[-1])

    Q_l = tf.range(Q_shape[-1], dtype=tf.int32)
    step = max_length // Q_shape[-1]
    Q_l = Q_l * step

    K_l = tf.range(K_shape[-1], dtype=tf.int32)
    step = max_length // K_shape[-1]
    K_l = K_l * step

    return (Q_l[..., tf.newaxis], K_l[..., tf.newaxis]), (Q_l, K_l)


class SyncModeScaleEnd(object):
  _SYNC_MODE = 'scale_end'

  def _compute_locations(self, Q_shape, K_shape):
    max_length = tf.maximum(Q_shape[-1], K_shape[-1])
    Q_l = tf.range(Q_shape[-1], dtype=tf.int32)
    step = max_length // Q_shape[-1]
    Q_l = (Q_l+1) * step - 1

    K_l = tf.range(K_shape[-1], dtype=tf.int32)
    step = max_length // K_shape[-1]
    K_l = (K_l+1) * step - 1

    return (Q_l[..., tf.newaxis], K_l[..., tf.newaxis]), (Q_l, K_l)


class Test(object):
  class Base(test_base.Test.Base):
    _SEQUENCE_DIMS = 1

    _SHAPE_DESC_TABLE = {
      tf.float16: ShapeDesc(tf.constant([1, 8, 8, 256], dtype=tf.int32),
                            tf.constant([1, 8, 32, 4096], dtype=tf.int32)),

      tf.float32: ShapeDesc(tf.constant([1, 8, 8, 256], dtype=tf.int32),
                            tf.constant([1, 8, 32, 2048], dtype=tf.int32)),

      tf.float64: ShapeDesc(tf.constant([1, 8, 8, 256], dtype=tf.int32),
                            tf.constant([1, 8, 32, 1024], dtype=tf.int32)),
    }


    def _vanilla_attention(self, Q, K, V, mask):
      Q_shape = tf.shape(Q)
      K_shape = tf.shape(K)
      logit = tf.einsum('bhcq,bhck->bhqk', Q, K) / tf.sqrt(tf.cast(Q_shape[-self._SEQUENCE_DIMS-1], dtype=Q.dtype))
      logit = tf.where(mask, logit, logit.dtype.min)
      p = tf.nn.softmax(logit, axis=-1)
      p = tf.where(mask, p, tf.constant(0.0, dtype=p.dtype))
      return tf.einsum('bhqk,bhck->bhcq', p, V), p

  class FullAttentionBase(test_base.VanillaFullnessPolicy, Base):

    def _flash_attention(self, Q, K, V):
      return flash_attention.full_1d(Q, K, V, sync_mode=self._SYNC_MODE, returning_l_m=True)

  class CausalAttentionBase(test_base.VanillaCausalityPolicy, Base):

    def _flash_attention(self, Q, K, V):
      return flash_attention.causal_1d(Q, K, V, sync_mode=self._SYNC_MODE, returning_l_m=True)

  class LocalAttentionBase(test_base.VanillaLocalPolicy, Base):

    def _flash_attention(self, Q, K, V):
      return flash_attention.local_1d(Q, K, V, window_size=self._window_size, log2_stride_size=self._log2_stride_size, is_causal=self._IS_CAUSAL, sync_mode=self._SYNC_MODE, returning_l_m=True)

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
  python -m flash_attention.tests.test_1d TestGroup.list

  To launch all tests at once:
  python -m flash_attention.tests.test_1d TestGroup.verify

  To launch one specific test, for example the test named "FullAttentionSyncModeNoneFront":
  TESTCASE=FullAttentionSyncModeNoneFront python -m flash_attention.tests.test_1d TestGroup.verify

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

