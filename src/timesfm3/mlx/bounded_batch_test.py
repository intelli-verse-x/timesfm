"""MLX bounded batching without pretrained checkpoint downloads."""
import unittest
from unittest import mock

import mlx.core as mx
import numpy as np

from .configs import TimesFM3MlxConfig
from .model import TimesFM3Mlx
from .timesfm3_forecaster import ModelConfig, TimesFM3Forecaster


class BoundedBatchTest(unittest.TestCase):
  def forecaster(self, batch_size):
    forecaster = TimesFM3Forecaster.__new__(TimesFM3Forecaster)
    forecaster.config = ModelConfig(per_core_batch_size=batch_size)
    forecaster.model = TimesFM3Mlx(
      TimesFM3MlxConfig(model_dims=32, hidden_dims=32, num_layers=1, num_heads=4),
      compile=False,
    )
    return forecaster

  def test_batch_limit_preserves_order_and_matches_single_forecasts(self):
    mx.random.seed(7)
    forecaster = self.forecaster(2)
    contexts = [np.sin(np.linspace(i, i + 5, n)).astype(np.float32)
                for i, n in enumerate([64, 96, 64, 64, 96, 64, 64])]
    ids = [f"series-{i}" for i in range(len(contexts))]
    with mock.patch.object(forecaster.model, "decode", wraps=forecaster.model.decode) as decode:
      outputs = list(forecaster.predict_batch(contexts, horizon=8, ts_ids=ids,
                                             return_quantiles=True))
      batch_sizes = [call.args[0].shape[0] for call in decode.call_args_list]
      self.assertTrue(all(call.args[1] == 64 for call in decode.call_args_list))
    self.assertEqual(batch_sizes, [2, 2, 1, 2])
    self.assertEqual([output.ts_id for output in outputs], ids)
    for context, output in zip(contexts, outputs):
      self.assertEqual(output.forecast.shape, (8,))
      self.assertEqual(output.quantiles.shape[0], 8)
      single = forecaster.predict(context, horizon=8, return_quantiles=True)
      np.testing.assert_allclose(output.forecast, single.forecast, atol=1e-4)
      np.testing.assert_allclose(output.quantiles, single.quantiles, atol=1e-4)

  def test_nonpositive_batch_size_is_rejected(self):
    for batch_size in (0, -1):
      with self.subTest(batch_size=batch_size):
        forecaster = self.forecaster(batch_size)
        with self.assertRaisesRegex(ValueError, "per_core_batch_size must be positive"):
          list(forecaster.predict_batch([np.ones(64, dtype=np.float32)], horizon=8))


  def test_compiled_frozen_stats_matches_eager_across_context_shapes(self):
    # Exercise the new freeze_after scalar through mx.compile, including a
    # cached shape revisited after a different context boundary. Shared tiny
    # weights isolate compilation from checkpoint provenance or downloads.
    for frozen in (False, True):
      with self.subTest(frozen=frozen):
        mx.random.seed(11)
        model = TimesFM3Mlx(
          TimesFM3MlxConfig(
            input_patch_len=8, output_patch_len=16,
            model_dims=32, hidden_dims=32, num_layers=1, num_heads=4,
            use_frozen_running_stats=frozen,
          ),
          compile=False,
        )
        for context_length in (32, 48, 32):
          target = mx.array(np.sin(np.arange(context_length) / 5).astype(np.float32))[None, None]
          future = mx.array(np.linspace(-3, 12, context_length + 8).astype(np.float32))[None, None]
          model._use_compile = False
          eager = np.array(model.decode(target, 8, past_future_covariates=future))
          model._use_compile = True
          compiled = np.array(model.decode(target, 8, past_future_covariates=future))
          self.assertIsNotNone(model._compiled_forward)
          self.assertEqual(compiled.shape, eager.shape)
          self.assertTrue(np.isfinite(compiled).all())
          np.testing.assert_allclose(compiled, eager, rtol=1e-4, atol=1e-4)
