"""Exercise the actual 2.5 compile/decode wrapper with an inert CPU decoder.

No checkpoint, GPU, remote model, application credential or provider is used.
The stub isolates channel selection; it does not establish forecasting quality.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from timesfm.configs import ForecastConfig
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch


def wrapper():
  model = SimpleNamespace(
    device_count=1, p=2, o=4, os=4, q=10, device=torch.device("cpu"),
    config=SimpleNamespace(context_limit=16),
  )

  def decode(horizon, inputs, masks):
    batch = inputs.shape[0]
    # Distinct channels and timesteps reveal index/order and backcast mistakes.
    values = torch.arange(80, dtype=torch.float32).reshape(1, 2, 4, 10)
    return values.repeat(batch, 1, 1, 1), torch.zeros(batch, 4, 10), None

  model.decode = decode
  instance = object.__new__(TimesFM_2p5_200M_torch)
  instance.model = model
  return instance


def compile_wrapper(instance, selection, backcast):
  instance.compile(ForecastConfig(
    max_context=4, max_horizon=4, per_core_batch_size=1,
    normalize_inputs=False, force_flip_invariance=False,
    infer_is_positive=False, fix_quantile_crossing=False,
    use_continuous_quantile_head=False, return_backcast=backcast,
    output_quantiles=selection,
  ))


@pytest.mark.parametrize("selection,channels", [
  (None, list(range(10))),
  ((0.1, 0.5, 0.9), [1, 5, 9]),
  ((0.9, 0.1), [9, 1]),
  ((), []),
])
@pytest.mark.parametrize("backcast", [False, True])
def test_requested_channels_preserve_median_point_and_backcast(selection, channels, backcast):
  instance = wrapper()
  compile_wrapper(instance, selection, backcast)
  points, quantiles = instance.compiled_decode(3, [[1., 2., 3., 4.]], [[False] * 4])
  expected = np.arange(80, dtype=np.float32).reshape(2, 4, 10)
  complete = expected[-1, :3]
  if backcast:
    complete = np.concatenate([expected[0, :2], complete], axis=0)
  np.testing.assert_array_equal(points[0], complete[:, 5])
  np.testing.assert_array_equal(quantiles[0], complete[:, channels])
  assert quantiles.shape == (1, 5 if backcast else 3, len(channels))


@pytest.mark.parametrize("selection", [(0.0,), (1.0,), (0.15,), (float("nan"),)])
def test_unsupported_quantile_is_rejected_before_a_decoder_is_installed(selection):
  instance = wrapper()
  with pytest.raises(ValueError, match="output_quantiles"):
    compile_wrapper(instance, selection, False)
  assert instance.compiled_decode is None
