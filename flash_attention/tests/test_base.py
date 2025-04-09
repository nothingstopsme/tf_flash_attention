import tensorflow as tf
import numpy as np
import collections, os

random_seed = 1234

ShapeDesc = collections.namedtuple('ShapeDesc', ('min', 'max'))

SyncClasses = collections.namedtuple('SyncClasses',
                                    ('none_front', 'scale_front', 'scale_end'))

AttentionClasses = collections.namedtuple('AttentionClasses',
                                    ('full', 'causal',
                                      'local', 'local_stride',
                                      'local_causal', 'local_stride_causal'))
class _VanillaPolicy(object):

  def _compute_diff(self, Q_l, K_l, has_location_dim):
    dim_offset = int(has_location_dim)

    Q_l_rank = Q_l.ndim - dim_offset
    K_l_rank = K_l.ndim - dim_offset
    for _ in range(K_l_rank):
      Q_l = tf.expand_dims(Q_l, axis=-1-dim_offset)

    for _ in range(Q_l_rank):
      K_l = tf.expand_dims(K_l, axis=0)

    diff = (Q_l - K_l)[tf.newaxis, ...]
    return diff


class VanillaFullnessPolicy(_VanillaPolicy):
  def _generate_mask(self, coords, indices):
    diff = self._compute_diff(*indices, has_location_dim=False)
    return tf.ones_like(diff) > 0

class VanillaCausalityPolicy(_VanillaPolicy):
  def _generate_mask(self, coords, indices):
    diff = self._compute_diff(*indices, has_location_dim=False)
    return diff >= 0

class VanillaLocalPolicy(_VanillaPolicy):
  _STRIDE_SIZE_ABOVE_1 = False
  _IS_CAUSAL = False

  def _generate_mask(self, coords, indices):
    diff = tf.abs(self._compute_diff(*coords, has_location_dim=True))
    if not self._IS_CAUSAL:
      predicator = tf.constant(True)
    else:
      predicator = self._compute_diff(*indices, has_location_dim=False) >= 0

    self._window_size = max(diff.shape)
    if self._STRIDE_SIZE_ABOVE_1:
      self._log2_stride_size = int(np.log2(self._window_size))
    else:
      self._log2_stride_size = 0

    local_stride = 2**self._log2_stride_size

    if local_stride > 1:
      return tf.logical_and(predicator,
                tf.reduce_all(tf.logical_and((diff % local_stride) == 0,
                          diff // local_stride < self._window_size), axis=-1))
    else:
      return tf.logical_and(predicator, tf.reduce_all(diff < self._window_size, axis=-1))


def show_benchmark_report(report):
  text_alignment_offset = 28
  for test_name, self_report in report.items():
    for dtype, type_report in self_report.items():
      for direction, (vanilla_report, flash_report) in type_report.items():

        print(f'{test_name}: {direction}, {dtype}')
        print(f'\033[{text_alignment_offset}Cvanilla\r\033[{text_alignment_offset*2}Cflash')
        print(
          (
            f'wall time: '
            f'\r\033[{text_alignment_offset}C{vanilla_report["wall_time"]}'
            f'\r\033[{text_alignment_offset*2}C{flash_report["wall_time"]}'
          )
        )
        print(
          (
            f'max # of bytes (GPU): '
            f'\r\033[{text_alignment_offset}C{vanilla_report["extras"]["allocator_maximum_num_bytes_GPU_0_bfc"]}'
            f'\r\033[{text_alignment_offset*2}C{flash_report["extras"]["allocator_maximum_num_bytes_GPU_0_bfc"]}'
          )
        )
        print('\n')


class Test(object):
  class Base(tf.test.TestCase, tf.test.Benchmark):
    _SEQUENCE_DIMS = None
    _SHAPE_DESC_TABLE = None
    _SYNC_MODE = None

    _RUNS = 20
    _BURNING_RUNS = 6

    def _compute_locations(self, Q_shape, K_shape):
      '''
      This function should return a tuple containing 2 elements:
      element 0 is (coordinates of entries in Q, coordinates of entries in K),
      and element 1 (indices of entries in Q flattend in row-major order, indices of entries in K flattend in row-major order)
      '''
      return super()._compute_locations(Q_shape, K_shape)

    def _generate_mask(self, *args):
      return super()._generate_mask(*args)

    def _vanilla_attention(self, Q, K, V, mask):
      raise NotImplemented('This function should be overridden by derived classes')

    def _flash_attention(self, Q, K, V):
      raise NotImplemented('This function should be overridden by derived classes')

    def _compute_alibi_bias(self, num_of_heads, Q_l, K_l, dtype):
      Q_shape = tf.shape(Q_l)
      K_shape = tf.shape(K_l)
      dimensions = tf.reduce_max(tf.stack((Q_shape, K_shape), axis=-1), axis=-1)
      num_of_entries = tf.cast(tf.reduce_prod(dimensions), dtype=dtype)

      all_coords = tf.stack(tf.meshgrid(*(tf.range(d, dtype=tf.int32) for d in dimensions), indexing='ij'), axis=-1)
      all_coords = tf.reshape(all_coords, (-1, Q_l.ndim))

      Q_coords = tf.gather(all_coords, Q_l, axis=0)
      K_coords = tf.gather(all_coords, K_l, axis=0)

      Q_coords = tf.reshape(Q_coords, tf.concat(([1], tf.shape(Q_coords)[:-1], [1]*K_l.ndim, tf.shape(Q_coords)[-1:]), axis=-1))
      K_coords = tf.reshape(K_coords, tf.concat(([1], [1]*Q_l.ndim, tf.shape(K_coords)), axis=-1))

      head_index = tf.cast(tf.reshape(tf.range(1, num_of_heads+1, dtype=tf.int32), (-1,)+(1,)*(Q_coords.ndim-2)), dtype=dtype)

      dist = tf.sqrt(tf.reduce_sum((tf.cast(Q_coords - K_coords, dtype=dtype) / num_of_entries)**2, axis=-1))

      bias = -tf.exp(tf.maximum(tf.cast(-8 / num_of_heads, dtype=dtype), tf.constant(-1.0, dtype=dtype)) * head_index) * dist * num_of_entries

      return bias

    def _generate_random_shape(self, min_shape, max_shape, even_seq_num):
      random_shape = tf.random.uniform(tf.shape(min_shape), seed=random_seed) * tf.cast(max_shape-min_shape+1, dtype=tf.float32)
      random_shape = tf.cast(random_shape, dtype=tf.int32) + min_shape

      if even_seq_num:
        random_shape = tf.concat((random_shape[:-1], random_shape[-1:] // 2 * 2), axis=-1)

      return random_shape

    def _generate_test_data(self, dtype, shape_gen):
      shape_desc = self._SHAPE_DESC_TABLE[dtype]
      if shape_gen == 'random':
        Q_shape = K_shape = V_shape = self._generate_random_shape(shape_desc.min, shape_desc.max, dtype==tf.float16)

        alt_seq_shape = self._generate_random_shape(shape_desc.min[-self._SEQUENCE_DIMS:], shape_desc.max[-self._SEQUENCE_DIMS:], dtype==tf.float16)

        Q_shape = tf.concat((Q_shape[:-self._SEQUENCE_DIMS], alt_seq_shape), axis=-1)

      elif shape_gen == 'max':
        Q_shape = K_shape = V_shape = shape_desc.max

      else:
        raise ValueError(f'Unsupported shape_gen: {shape_gen}')

      dO_shape = tf.concat((Q_shape[:-self._SEQUENCE_DIMS-1], V_shape[-self._SEQUENCE_DIMS-1:-self._SEQUENCE_DIMS], Q_shape[-self._SEQUENCE_DIMS:]), axis=-1)

      Q = tf.random.uniform(Q_shape, -2.0, 2.0, dtype=dtype, seed=random_seed)
      K = tf.random.uniform(K_shape, -2.0, 2.0, dtype=dtype, seed=random_seed)
      V = tf.random.uniform(V_shape, -2.0, 2.0, dtype=dtype, seed=random_seed)
      dO = tf.random.uniform(dO_shape, -2.0, 2.0, dtype=dtype, seed=random_seed)

      location_info = self._compute_locations(Q_shape, K_shape)
      mask = self._generate_mask(*location_info)

      return Q, K, V, mask, dO

    def verify(self):
      for dtype in (tf.float16, tf.float32, tf.float64):
        for _ in range(self._RUNS):
          Q, K, V, mask, dO = self._generate_test_data(dtype, 'random')

          with tf.GradientTape(persistent=True) as tape:
            tape.watch(Q)
            tape.watch(K)
            tape.watch(V)

            vanilla_forward_results = self._vanilla_attention(Q, K, V, mask)
            flash_forward_results = self._flash_attention(Q, K, V)

          vanilla_backward_results = tape.gradient(vanilla_forward_results[0], (Q, K, V), dO)
          flash_backward_results = tape.gradient(flash_forward_results[0], (Q, K, V), dO)

          del tape

          num_of_K_entries = tf.cast(tf.reduce_prod(tf.shape(K)[-self._SEQUENCE_DIMS:]), dtype=dtype)
          num_of_Q_entries = tf.cast(tf.reduce_prod(tf.shape(Q)[-self._SEQUENCE_DIMS:]), dtype=dtype)

          # Only comparing the attention output O
          vanilla = vanilla_forward_results[0]
          flash = flash_forward_results[0]

          # Conducting normalisation to account for the accumulation of precision errors
          normalised_error = tf.abs(vanilla - flash) / num_of_K_entries
          self.assertAllCloseAccordingToType(normalised_error, tf.zeros_like(normalised_error),
            msg=(
              f'{type(self).__name__}: forward, {dtype}, '
              f'random_seed = {random_seed}, '
              f'Q = {Q.shape}, K = {K.shape}, V = {V.shape}'
            ))

          for vanilla, flash, normaliser in zip(vanilla_backward_results, flash_backward_results,
                                    (num_of_K_entries, num_of_Q_entries, num_of_Q_entries)):
            # Conducting normalisation to account for the accumulation of precision errors
            normalised_error = tf.abs(vanilla - flash) / normaliser
            self.assertAllCloseAccordingToType(normalised_error, tf.zeros_like(normalised_error),
              msg=(
                f'{type(self).__name__}: backward, {dtype}, '
                f'random_seed = {random_seed}, '
                f'Q = {Q.shape}, K = {K.shape}, V = {V.shape}'
              ))

    def report_benchmark(self, **kwargs):
      # Overriding this function to prevent the benchmark report from being shown automatically;
      # instead manually calling _print_benchmark_report() after each benchmark run
      pass

    def benchmark(self, report=None):
      if report is None:
        show_report = True
        report = {}
      else:
        show_report = False

      self_report = report[type(self).__name__] = {}
      for dtype in (tf.float16, tf.float32, tf.float64):
        Q, K, V, mask, dO = self._generate_test_data(dtype, 'max')
        vanilla_forward_results = self._vanilla_attention(Q, K, V, mask)
        flash_forward_results = self._flash_attention(Q, K, V)

        Q, K, V, mask, dO = tuple(tensor.numpy() for tensor in (Q, K, V, mask, dO))
        vanilla_forward_results = tuple(tensor.numpy() for tensor in vanilla_forward_results)
        flash_forward_results = tuple(tensor.numpy() for tensor in flash_forward_results)

        type_report = self_report[dtype] = {}
        with tf.Graph().as_default() as graph:
          Q_tensor = tf.compat.v1.placeholder(dtype=Q.dtype, shape=Q.shape)
          K_tensor = tf.compat.v1.placeholder(dtype=K.dtype, shape=K.shape)
          V_tensor = tf.compat.v1.placeholder(dtype=V.dtype, shape=V.shape)
          mask_tensor = tf.compat.v1.placeholder(dtype=mask.dtype, shape=mask.shape)
          dO_tensor = tf.compat.v1.placeholder(dtype=dO.dtype, shape=dO.shape)


          with tf.GradientTape(persistent=True) as tape:
            tape.watch(Q_tensor)
            tape.watch(K_tensor)
            tape.watch(V_tensor)

            vanilla_forward_result_tensors = self._vanilla_attention(Q_tensor, K_tensor, V_tensor, mask_tensor)
            flash_forward_result_tensors = self._flash_attention(Q_tensor, K_tensor, V_tensor)

          vanilla_backward_result_tensors = tape.gradient(vanilla_forward_result_tensors[0], [Q_tensor, K_tensor, V_tensor], dO_tensor)
          flash_backward_result_tensors = tape.gradient(flash_forward_result_tensors[0], [Q_tensor, K_tensor, V_tensor], dO_tensor)
          del tape


          with tf.compat.v1.Session(graph=graph) as sess:
            vanilla_feed_dict = {Q_tensor: Q, K_tensor: K, V_tensor: V,
                                  mask_tensor: mask}
            vanilla_report = super().run_op_benchmark(sess, vanilla_forward_result_tensors[0], feed_dict=vanilla_feed_dict, burn_iters=self._BURNING_RUNS, min_iters=self._RUNS)

          with tf.compat.v1.Session(graph=graph) as sess:
            flash_feed_dict = {Q_tensor: Q, K_tensor: K, V_tensor: V}
            flash_report = super().run_op_benchmark(sess, flash_forward_result_tensors[0], feed_dict=flash_feed_dict, burn_iters=self._BURNING_RUNS, min_iters=self._RUNS)

          type_report['forward'] = (vanilla_report, flash_report)



          with tf.compat.v1.Session(graph=graph) as sess:
            vanilla_feed_dict = {Q_tensor: Q, K_tensor: K, V_tensor: V,
                                  mask_tensor: mask, dO_tensor: dO}
            vanilla_feed_dict.update({
              key: value for key, value in zip(vanilla_forward_result_tensors, vanilla_forward_results)
            })
            vanilla_report = super().run_op_benchmark(sess, vanilla_backward_result_tensors, feed_dict=vanilla_feed_dict, burn_iters=self._BURNING_RUNS, min_iters=self._RUNS)

          with tf.compat.v1.Session(graph=graph) as sess:
            flash_feed_dict = {Q_tensor: Q, K_tensor: K, V_tensor: V, dO_tensor: dO}
            flash_feed_dict.update({
              key: value for key, value in zip(flash_forward_result_tensors, flash_forward_results)
            })
            flash_report = super().run_op_benchmark(sess, flash_backward_result_tensors, feed_dict=flash_feed_dict, burn_iters=self._BURNING_RUNS, min_iters=self._RUNS)

          type_report['backward'] = (vanilla_report, flash_report)


      if show_report:
        show_benchmark_report(report)



class TestGroup(tf.test.TestCase):

  def __init__(self, sync_classes, attention_classes, *args, **kwargs):
    super().__init__(*args, **kwargs)

    class FullAttentionSyncModeNoneFront(sync_classes.none_front, attention_classes.full):
      pass

    class CausalAttentionSyncModeNoneFront(sync_classes.none_front, attention_classes.causal):
      pass

    class CausalAttentionSyncModeScaleFront(sync_classes.scale_front, attention_classes.causal):
      pass

    class CausalAttentionSyncModeScaleEnd(sync_classes.scale_end, attention_classes.causal):
      pass

    class LocalAttentionSyncModeNoneFront(sync_classes.none_front, attention_classes.local):
      pass

    class LocalAttentionSyncModeScaleFront(sync_classes.scale_front, attention_classes.local):
      pass

    class LocalAttentionSyncModeScaleEnd(sync_classes.scale_end, attention_classes.local):
      pass

    class LocalStrideAttentionSyncModeNoneFront(sync_classes.none_front, attention_classes.local_stride):
      pass

    class LocalStrideAttentionSyncModeScaleFront(sync_classes.scale_front, attention_classes.local_stride):
      pass

    class LocalStrideAttentionSyncModeScaleEnd(sync_classes.scale_end, attention_classes.local_stride):
      pass

    class LocalAndCausalAttentionSyncModeNoneFront(sync_classes.none_front, attention_classes.local_causal):
      pass

    class LocalAndCausalAttentionSyncModeScaleFront(sync_classes.scale_front, attention_classes.local_causal):
      pass

    class LocalAndCausalAttentionSyncModeScaleEnd(sync_classes.scale_end, attention_classes.local_causal):
      pass

    class LocalStrideAndCausalAttentionSyncModeNoneFront(sync_classes.none_front, attention_classes.local_stride_causal):
      pass

    class LocalStrideAndCausalAttentionSyncModeScaleFront(sync_classes.scale_front, attention_classes.local_stride_causal):
      pass

    class LocalStrideAndCausalAttentionSyncModeScaleEnd(sync_classes.scale_end, attention_classes.local_stride_causal):
      pass



    self._testcases = dict(
                  FullAttentionSyncModeNoneFront=FullAttentionSyncModeNoneFront(),

                  CausalAttentionSyncModeScaleFront=CausalAttentionSyncModeScaleFront(),
                  CausalAttentionSyncModeScaleEnd=CausalAttentionSyncModeScaleEnd(),

                  LocalAttentionSyncModeNoneFront=LocalAttentionSyncModeNoneFront(),
                  LocalAttentionSyncModeScaleFront=LocalAttentionSyncModeScaleFront(),
                  LocalAttentionSyncModeScaleEnd=LocalAttentionSyncModeScaleEnd(),

                  LocalStrideAttentionSyncModeNoneFront=LocalStrideAttentionSyncModeNoneFront(),
                  LocalStrideAttentionSyncModeScaleFront=LocalStrideAttentionSyncModeScaleFront(),
                  LocalStrideAttentionSyncModeScaleEnd=LocalStrideAttentionSyncModeScaleEnd(),

                  LocalAndCausalAttentionSyncModeNoneFront=LocalAndCausalAttentionSyncModeNoneFront(),
                  LocalAndCausalAttentionSyncModeScaleFront=LocalAndCausalAttentionSyncModeScaleFront(),
                  LocalAndCausalAttentionSyncModeScaleEnd=LocalAndCausalAttentionSyncModeScaleEnd(),

                  LocalStrideAndCausalAttentionSyncModeNoneFront=LocalStrideAndCausalAttentionSyncModeNoneFront(),
                  LocalStrideAndCausalAttentionSyncModeScaleFront=LocalStrideAndCausalAttentionSyncModeScaleFront(),
                  LocalStrideAndCausalAttentionSyncModeScaleEnd=LocalStrideAndCausalAttentionSyncModeScaleEnd(),
                )

  def _prepare_testcases(self):
    name = os.environ.get('TESTCASE', 'all')
    if name != 'all':
      return {name:self._testcases[name]}
    else:
      return self._testcases

  def list(self):
    print('Available testcases:')
    for name in self._prepare_testcases().keys():
      print(name)

  def verify(self):
    for name, testcase in self._prepare_testcases().items():
      print(f'Verifying {name}')
      testcase.verify()

  def benchmark(self):
    report = {}
    for name, testcase in self._prepare_testcases().items():
      print(f'Benchmarking {name}')
      testcase.benchmark(report)

    show_benchmark_report(report)




