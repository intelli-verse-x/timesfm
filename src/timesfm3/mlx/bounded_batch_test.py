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
    self.assertEqual(batch_sizes, [2, 2, 1, 2])
    self.assertEqual([output.ts_id for output in outputs], ids)
    for context, output in zip(contexts, outputs):
      single = forecaster.predict(context, horizon=8, return_quantiles=True)
      np.testing.assert_allclose(output.forecast, single.forecast, atol=1e-4)
      np.testing.assert_allclose(output.quantiles, single.quantiles, atol=1e-4)

  def test_nonpositive_batch_size_is_rejected(self):
    for batch_size in (0, -1):
      with self.subTest(batch_size=batch_size):
        forecaster = self.forecaster(batch_size)
        with self.assertRaisesRegex(ValueError, "per_core_batch_size must be positive"):
          list(forecaster.predict_batch([np.ones(64, dtype=np.float32)], horizon=8))
