"""
Unit tests for attention mechanisms.

Tests basic functionality and interface compliance.
"""

import pytest
import torch
import sys
import runpy
import textwrap
import numpy as np
from pathlib import Path
from typing import Any, cast
import matplotlib

matplotlib.use("Agg")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from attention import (
    CrossModalAttention,
    TemporalAttention,
    PairwiseModalityAttention,
    visualize_attention,
)


class TestCrossModalAttention:
    """Test CrossModalAttention module."""

    @pytest.fixture
    def attention_params(self):
        return {
            "query_dim": 512,
            "key_dim": 64,
            "hidden_dim": 256,
            "num_heads": 4,
        }

    @pytest.fixture
    def batch_size(self):
        return 4

    def test_output_shape(self, attention_params, batch_size):
        """Test CrossModalAttention output shape."""
        try:
            attn = CrossModalAttention(**attention_params)

            query = torch.randn(batch_size, attention_params["query_dim"])
            key = torch.randn(batch_size, attention_params["key_dim"])
            value = torch.randn(batch_size, attention_params["key_dim"])

            attended, weights = attn(query, key, value)

            assert attended.shape == (
                batch_size,
                attention_params["hidden_dim"],
            ), (
                f"Expected shape ({batch_size}, {attention_params['hidden_dim']}), got {attended.shape}"
            )
            assert weights is not None, "Attention weights should be returned"
        except NotImplementedError:
            pytest.skip("CrossModalAttention not implemented yet")

    def test_with_mask(self, attention_params, batch_size):
        """Test CrossModalAttention with mask."""
        try:
            attn = CrossModalAttention(**attention_params)

            query = torch.randn(batch_size, attention_params["query_dim"])
            key = torch.randn(batch_size, attention_params["key_dim"])
            value = torch.randn(batch_size, attention_params["key_dim"])
            mask = torch.tensor(
                [1, 1, 0, 1], dtype=torch.float
            )  # Third key masked

            attended, weights = attn(query, key, value, mask)

            assert not torch.isnan(attended).any(), (
                "Output contains NaN with mask"
            )
        except NotImplementedError:
            pytest.skip("CrossModalAttention not implemented yet")

    def test_gradient_flow(self, attention_params):
        """Test gradient flow through CrossModalAttention."""
        try:
            attn = CrossModalAttention(**attention_params)

            query = torch.randn(
                2, attention_params["query_dim"], requires_grad=True
            )
            key = torch.randn(
                2, attention_params["key_dim"], requires_grad=True
            )
            value = torch.randn(
                2, attention_params["key_dim"], requires_grad=True
            )

            attended, _ = attn(query, key, value)
            loss = attended.sum()
            loss.backward()

            assert query.grad is not None, "No gradient for query"
            assert key.grad is not None, "No gradient for key"
            assert value.grad is not None, "No gradient for value"
        except NotImplementedError:
            pytest.skip("CrossModalAttention not implemented yet")


class TestTemporalAttention:
    """Test TemporalAttention module."""

    @pytest.fixture
    def attention_params(self):
        return {"feature_dim": 128, "hidden_dim": 256, "num_heads": 4}

    @pytest.fixture
    def sequence_data(self):
        batch_size = 4
        seq_len = 10
        feature_dim = 128
        return torch.randn(batch_size, seq_len, feature_dim)

    def test_output_shape(self, attention_params, sequence_data):
        """Test TemporalAttention output shape."""
        try:
            attn = TemporalAttention(**attention_params)
            attended_seq, weights = attn(sequence_data)

            batch_size, seq_len, _ = sequence_data.shape
            expected_shape = (
                batch_size,
                seq_len,
                attention_params["hidden_dim"],
            )

            assert attended_seq.shape == expected_shape, (
                f"Expected shape {expected_shape}, got {attended_seq.shape}"
            )
            assert weights is not None, "Attention weights should be returned"
        except NotImplementedError:
            pytest.skip("TemporalAttention not implemented yet")

    def test_with_mask(self, attention_params):
        """Test TemporalAttention with variable-length sequences."""
        try:
            attn = TemporalAttention(**attention_params)

            batch_size = 2
            seq_len = 10
            sequence = torch.randn(
                batch_size, seq_len, attention_params["feature_dim"]
            )
            mask = torch.zeros(batch_size, seq_len)
            mask[0, :7] = 1
            mask[1, :5] = 1

            attended_seq, _ = attn(sequence, mask)
            assert not torch.isnan(attended_seq).any(), (
                "Output contains NaN with mask"
            )
        except NotImplementedError:
            pytest.skip("TemporalAttention not implemented yet")

    def test_mask_vector_zeroes_positions(self, attention_params):
        """Mask vectors should zero masked timesteps after attention."""
        try:
            attn = TemporalAttention(**attention_params)
            batch_size, seq_len = 2, 6
            sequence = torch.randn(
                batch_size, seq_len, attention_params["feature_dim"]
            )
            mask = torch.tensor([1, 0, 1, 0, 1, 1], dtype=torch.float32)

            attended, _ = attn(sequence, mask)
            collapsed = attended.squeeze(0).squeeze(0)
            zero_positions = mask == 0
            assert torch.allclose(
                collapsed[:, zero_positions, :],
                torch.zeros_like(collapsed[:, zero_positions, :]),
            )
        except NotImplementedError:
            pytest.skip("TemporalAttention not implemented yet")

    def test_pool_sequence_valid_and_invalid(self, attention_params):
        """Ensure pooling validates shapes and returns normalized summaries."""
        try:
            attn = TemporalAttention(**attention_params)
            batch_size, seq_len = 3, 4
            hidden_dim = attention_params["hidden_dim"]
            sequence = torch.randn(batch_size, seq_len, hidden_dim)
            weights = torch.rand(batch_size, attn.num_heads, seq_len, seq_len)

            pooled = attn.pool_sequence(sequence, weights)
            assert pooled.shape == (batch_size, hidden_dim)

            with pytest.raises(ValueError):
                attn.pool_sequence(sequence, weights.mean(dim=1))
        except NotImplementedError:
            pytest.skip("TemporalAttention not implemented yet")


class TestPairwiseModalityAttention:
    """Test PairwiseModalityAttention module."""

    @pytest.fixture
    def modality_dims(self):
        return {"video": 512, "audio": 128, "imu": 64}

    @pytest.fixture
    def modality_features(self, modality_dims):
        batch_size = 4
        return {
            modality: torch.randn(batch_size, dim)
            for modality, dim in modality_dims.items()
        }

    def test_output_structure(self, modality_dims, modality_features):
        """Test PairwiseModalityAttention output structure."""
        try:
            attn = PairwiseModalityAttention(
                modality_dims=modality_dims, hidden_dim=256, num_heads=4
            )

            attended_features, attention_maps = attn(modality_features)

            assert isinstance(attended_features, dict), (
                "Should return dict of features"
            )
            assert isinstance(attention_maps, dict), (
                "Should return dict of attention maps"
            )
            assert len(attended_features) == len(modality_dims), (
                "Should have attended features for each modality"
            )
        except NotImplementedError:
            pytest.skip("PairwiseModalityAttention not implemented yet")

    def test_requires_modalities(self):
        """Ensure missing modality configuration raises a helpful error."""
        attn = PairwiseModalityAttention(modality_dims={})
        with pytest.raises(ValueError):
            attn({}, modality_mask=None)

    def test_modality_mask_and_missing_layer_guard(
        self, modality_dims, modality_features
    ):
        """Verify masks zero unavailable modalities and missing layers are skipped."""
        attn = PairwiseModalityAttention(
            modality_dims=modality_dims, hidden_dim=64, num_heads=2
        )
        removed_key = f"{attn.modality_names[0]}_to_{attn.modality_names[1]}"
        del attn.attention_layers[removed_key]

        batch_size = next(iter(modality_features.values())).shape[0]
        modality_mask = torch.tensor(
            [[True, False, True]] * batch_size, dtype=torch.bool
        )

        attended, maps = attn(modality_features, modality_mask=modality_mask)

        assert removed_key not in maps
        masked_idx = attn.modality_names.index("audio")
        assert torch.allclose(
            attended["audio"][modality_mask[:, masked_idx] == 0],
            torch.zeros_like(
                attended["audio"][modality_mask[:, masked_idx] == 0]
            ),
        )


class TestVisualizeAttention:
    """Test visualize_attention utility."""

    @pytest.fixture
    def modality_names(self):
        return ["video", "audio", "imu", "text", "depth"]

    def test_visualize_attention_reduces_three_dim_tensor(
        self, modality_names, tmp_path, monkeypatch
    ):
        """Ensure 3D tensors are averaged across heads and saved via string paths."""

        class DummyFigure:
            def __init__(self):
                self.saved_path = None

            def savefig(self, path, **kwargs):
                self.saved_path = Path(path)

        class DummyAxes:
            def __init__(self):
                self.imshow_data = None

            def imshow(self, data, **kwargs):
                self.imshow_data = data
                return object()

            def set_xticks(self, *args, **kwargs):
                return None

            def set_yticks(self, *args, **kwargs):
                return None

            def set_xticklabels(self, *args, **kwargs):
                return None

            def set_yticklabels(self, *args, **kwargs):
                return None

            def set_xlabel(self, *args, **kwargs):
                return None

            def set_ylabel(self, *args, **kwargs):
                return None

            def set_title(self, *args, **kwargs):
                return None

        dummy_fig = DummyFigure()
        dummy_ax = DummyAxes()
        close_calls: list[object] = []

        monkeypatch.setattr(
            "matplotlib.pyplot.subplots",
            lambda figsize=None: (dummy_fig, dummy_ax),
        )
        monkeypatch.setattr(
            "matplotlib.pyplot.colorbar", lambda *args, **kwargs: None
        )
        monkeypatch.setattr("matplotlib.pyplot.tight_layout", lambda: None)
        monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
        monkeypatch.setattr(
            "matplotlib.pyplot.close", lambda fig: close_calls.append(fig)
        )

        tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        expected_heatmap = tensor.mean(dim=0).numpy()
        save_path = tmp_path / "nested" / "attention.png"

        visualize_attention(tensor, modality_names, save_path=str(save_path))

        assert dummy_ax.imshow_data is not None, (
            "Expected heatmap passed to imshow"
        )
        assert dummy_ax.imshow_data.shape == expected_heatmap.shape
        assert torch.allclose(
            torch.as_tensor(dummy_ax.imshow_data),
            torch.as_tensor(expected_heatmap),
        )
        assert dummy_fig.saved_path == save_path
        assert close_calls == [dummy_fig], (
            "Figure should be closed after saving"
        )

    def test_visualize_attention_saves_file(
        self, modality_names, tmp_path, monkeypatch
    ):
        """Ensure visualize_attention saves figure and skips show when path provided."""
        show_called = False

        def fake_show() -> None:
            nonlocal show_called
            show_called = True

        monkeypatch.setattr("matplotlib.pyplot.show", fake_show)

        attention_weights = torch.rand(2, 3, 4, 5)
        save_path = tmp_path / "plots" / "attention.png"

        visualize_attention(
            attention_weights, modality_names, save_path=save_path
        )

        assert save_path.is_file(), (
            "Attention visualization should be saved to disk"
        )
        assert save_path.stat().st_size > 0, "Saved figure should not be empty"
        assert not show_called, (
            "plt.show should not be called when save_path is set"
        )

    def test_visualize_attention_scalar_triggers_show(
        self, modality_names, monkeypatch
    ):
        """Ensure scalar attention weights are normalized and trigger plotting."""
        show_calls: list[bool] = []

        def fake_show() -> None:
            show_calls.append(True)

        monkeypatch.setattr("matplotlib.pyplot.show", fake_show)

        visualize_attention(torch.tensor(0.5), modality_names[:2])

        assert show_calls == [True], (
            "plt.show should be called once when save_path is None"
        )

    def test_visualize_attention_accepts_sequence_input(
        self, modality_names, monkeypatch
    ):
        """Ensure non-tensor sequences are coerced and plotted."""
        show_calls: list[bool] = []

        def fake_show() -> None:
            show_calls.append(True)

        monkeypatch.setattr("matplotlib.pyplot.show", fake_show)

        visualize_attention(cast(Any, [0.2, 0.8, 0.0]), modality_names)

        assert show_calls, "Sequence input should still render via plt.show"

    def test_visualize_attention_expands_vector_heatmap(
        self, modality_names, tmp_path, monkeypatch
    ):
        """Heatmaps squeezed to 1D should be expanded back to matrix form."""
        orig_numpy = torch.Tensor.numpy

        def squeezed_numpy(self):
            return np.squeeze(orig_numpy(self))

        monkeypatch.setattr(
            torch.Tensor, "numpy", squeezed_numpy, raising=False
        )

        save_path = tmp_path / "expanded.png"
        extended_names = modality_names + ["lidar"]
        visualize_attention(
            torch.arange(6.0).reshape(1, 6), extended_names, save_path
        )
        assert save_path.exists()


def test_attention_main_runs(capsys):
    """Run module entrypoint to exercise __main__ coverage."""
    runpy.run_module("attention", run_name="__main__")
    output = capsys.readouterr().out
    assert "Testing attention mechanisms" in output


def test_attention_main_handles_errors(monkeypatch, capsys):
    """Ensure __main__ handles missing implementations gracefully."""
    import attention as attention_module

    lines = Path("src/attention.py").read_text().splitlines()
    start = next(
        idx
        for idx, line in enumerate(lines)
        if line.strip().startswith("# Simple test")
    )
    block = "\n".join(lines[start:])
    injected = "\n" * start + textwrap.dedent(block)

    compiled = compile(injected, "src/attention.py", "exec")
    scenarios = [
        (
            lambda self, *args, **kwargs: (_ for _ in ()).throw(
                NotImplementedError("stub")
            ),
            lambda self, *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("boom")
            ),
            ["not implemented yet", "temporalattention error"],
        ),
        (
            lambda self, *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("boom")
            ),
            lambda self, *args, **kwargs: (_ for _ in ()).throw(
                NotImplementedError("stub")
            ),
            ["crossmodalattention error", "not implemented yet"],
        ),
    ]

    for cross_fn, temporal_fn, expected in scenarios:
        monkeypatch.setattr(
            attention_module.CrossModalAttention,
            "forward",
            cross_fn,
            raising=False,
        )
        monkeypatch.setattr(
            attention_module.TemporalAttention,
            "forward",
            temporal_fn,
            raising=False,
        )
        namespace = dict(attention_module.__dict__)
        namespace["__name__"] = "__main__"
        capsys.readouterr()
        exec(compiled, namespace)
        output = capsys.readouterr().out.lower()
        for message in expected:
            assert message in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
