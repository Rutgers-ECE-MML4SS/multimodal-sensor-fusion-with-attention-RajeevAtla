"""
Unit tests for encoder modules.

Tests basic functionality and interface compliance.
"""

import pytest
import torch
import torch.nn as nn
import sys
import runpy
from pathlib import Path
from typing import Any, cast

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from encoders import SequenceEncoder, FrameEncoder, SimpleMLPEncoder, build_encoder


class TestSequenceEncoder:
    """Test SequenceEncoder module."""

    @pytest.fixture
    def encoder_params(self):
        return {"input_dim": 64, "hidden_dim": 128, "output_dim": 64, "num_layers": 2}

    @pytest.fixture
    def sequence_data(self):
        batch_size = 4
        seq_len = 100
        input_dim = 64
        return torch.randn(batch_size, seq_len, input_dim)

    def test_lstm_encoder_shape(self, encoder_params, sequence_data):
        """Test LSTM SequenceEncoder output shape."""
        try:
            encoder = SequenceEncoder(**encoder_params, encoder_type="lstm")
            output = encoder(sequence_data)

            batch_size = sequence_data.size(0)
            expected_shape = (batch_size, encoder_params["output_dim"])

            assert output.shape == expected_shape, (
                f"Expected shape {expected_shape}, got {output.shape}"
            )
            print("✓ LSTM SequenceEncoder shape test passed")
        except NotImplementedError:
            pytest.skip("LSTM SequenceEncoder not implemented yet")

    def test_gru_encoder_shape(self, encoder_params, sequence_data):
        """Test GRU SequenceEncoder output shape."""
        try:
            encoder = SequenceEncoder(**encoder_params, encoder_type="gru")
            output = encoder(sequence_data)

            batch_size = sequence_data.size(0)
            expected_shape = (batch_size, encoder_params["output_dim"])

            assert output.shape == expected_shape, (
                f"Expected shape {expected_shape}, got {output.shape}"
            )
            print("✓ GRU SequenceEncoder shape test passed")
        except NotImplementedError:
            pytest.skip("GRU SequenceEncoder not implemented yet")

    def test_cnn_encoder_shape(self, encoder_params, sequence_data):
        """Test CNN SequenceEncoder output shape."""
        try:
            encoder = SequenceEncoder(**encoder_params, encoder_type="cnn")
            output = encoder(sequence_data)

            batch_size = sequence_data.size(0)
            expected_shape = (batch_size, encoder_params["output_dim"])

            assert output.shape == expected_shape, (
                f"Expected shape {expected_shape}, got {output.shape}"
            )
            print("✓ CNN SequenceEncoder shape test passed")
        except NotImplementedError:
            pytest.skip("CNN SequenceEncoder not implemented yet")

    def test_sequence_encoder_requires_three_dim_input(self, encoder_params):
        """SequenceEncoder should validate input dimensionality."""
        encoder = SequenceEncoder(**encoder_params, encoder_type="lstm")
        invalid_input = torch.randn(encoder_params["input_dim"])
        with pytest.raises(ValueError, match="Expected 3D input sequence"):
            encoder(invalid_input)

    def test_sequence_encoder_unknown_type_raises(self, encoder_params):
        """Invalid encoder type should raise during initialization."""
        with pytest.raises(ValueError, match="Unknown encoder type"):
            SequenceEncoder(**encoder_params, encoder_type="invalid")

    def test_sequence_encoder_missing_rnn_raises(self, encoder_params):
        """Missing RNN module should produce a runtime error."""
        encoder = SequenceEncoder(**encoder_params, encoder_type="gru")
        cast(Any, encoder).rnn = None
        with pytest.raises(RuntimeError, match="RNN module not initialized"):
            encoder(torch.randn(2, 4, encoder_params["input_dim"]))

    def test_sequence_encoder_unsupported_forward_type(self, encoder_params):
        """Forward path should guard against unexpected encoder_type values."""
        encoder = SequenceEncoder(**encoder_params, encoder_type="lstm")
        cast(Any, encoder).encoder_type = "bogus"
        with pytest.raises(ValueError, match="Unsupported encoder type"):
            encoder(torch.randn(2, 4, encoder_params["input_dim"]))

    def test_cnn_missing_modules_raises(self, encoder_params, sequence_data):
        """Ensure CNN SequenceEncoder raises when required modules are missing."""
        try:
            encoder = SequenceEncoder(**encoder_params, encoder_type="cnn")
            cast(Any, encoder).conv_net = None
            cast(Any, encoder).pool = None

            with pytest.raises(RuntimeError, match="CNN modules not initialized"):
                encoder(sequence_data)
        except NotImplementedError:
            pytest.skip("CNN SequenceEncoder not implemented yet")

    def test_cnn_encoder_average_pooling_identity_modules(self):
        """Ensure CNN SequenceEncoder pools over the temporal axis as expected."""
        try:
            encoder = SequenceEncoder(
                input_dim=4,
                hidden_dim=4,
                output_dim=4,
                num_layers=1,
                encoder_type="cnn",
                dropout=0.0,
            )
            cast(Any, encoder).conv_net = nn.Identity()
            cast(Any, encoder).pool = nn.AdaptiveAvgPool1d(1)
            cast(Any, encoder).dropout_layer = nn.Identity()
            cast(Any, encoder).projection = nn.Identity()

            sequence = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
            output = encoder(sequence)

            expected = sequence.mean(dim=1)
            torch.testing.assert_close(output, expected)
        except NotImplementedError:
            pytest.skip("CNN SequenceEncoder not implemented yet")

    def test_variable_length_sequences(self, encoder_params):
        """Test SequenceEncoder with variable-length sequences."""
        try:
            encoder = SequenceEncoder(**encoder_params, encoder_type="lstm")

            batch_size = 2
            seq_len = 100
            sequence = torch.randn(batch_size, seq_len, encoder_params["input_dim"])
            lengths = torch.tensor([80, 60])  # Different actual lengths

            output = encoder(sequence, lengths)

            assert output.shape == (batch_size, encoder_params["output_dim"])
            assert not torch.isnan(output).any(), "Output contains NaN"
            print("✓ Variable-length sequence test passed")
        except NotImplementedError:
            pytest.skip("Variable-length handling not implemented yet")

    def test_transformer_encoder_applies_mask(self):
        """Ensure transformer SequenceEncoder applies padding mask during pooling."""
        try:
            encoder = SequenceEncoder(
                input_dim=4,
                hidden_dim=4,
                output_dim=4,
                num_layers=1,
                encoder_type="transformer",
                dropout=0.0,
            )

            captured_masks: list[torch.Tensor | None] = []

            class IdentityTransformer(nn.Module):
                def __init__(self):
                    super().__init__()

                def forward(
                    self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None
                ) -> torch.Tensor:
                    captured_masks.clear()
                    captured_masks.append(src_key_padding_mask)
                    time_offsets = (
                        torch.arange(x.size(1), device=x.device, dtype=x.dtype)
                        .view(1, -1, 1)
                        .expand_as(x)
                    )
                    return x + time_offsets

            transformer = IdentityTransformer()
            cast(Any, encoder).transformer = transformer
            cast(Any, encoder).input_projection = nn.Identity()
            cast(Any, encoder).dropout_layer = nn.Identity()
            cast(Any, encoder).projection = nn.Identity()

            sequence = torch.arange(40, dtype=torch.float32).view(2, 5, 4)
            lengths = torch.tensor([3, 5])

            output = encoder(sequence, lengths)

            assert captured_masks and captured_masks[0] is not None
            last_mask = captured_masks[0]
            assert last_mask is not None
            expected_mask = torch.tensor(
                [[False, False, False, True, True], [False, False, False, False, False]]
            )
            assert last_mask.shape == expected_mask.shape
            assert torch.equal(last_mask, expected_mask)

            expected_valid = (~expected_mask).unsqueeze(-1).float()
            time_offsets = (
                torch.arange(sequence.size(1), dtype=sequence.dtype)
                .view(1, -1, 1)
                .expand_as(sequence)
            )
            transformer_out = sequence + time_offsets
            expected = (transformer_out * expected_valid).sum(dim=1) / expected_valid.sum(
                dim=1
            ).clamp_min(1.0)

            torch.testing.assert_close(output, expected)
        except NotImplementedError:
            pytest.skip("Transformer SequenceEncoder not implemented yet")

    def test_transformer_encoder_without_lengths_uses_mean_pooling(self):
        """Transformer encoder should fall back to simple mean pooling when no mask is provided."""
        try:
            encoder = SequenceEncoder(
                input_dim=4,
                hidden_dim=4,
                output_dim=4,
                num_layers=1,
                encoder_type="transformer",
                dropout=0.0,
            )

            captured_masks: list[torch.Tensor | None] = []

            class ShiftTransformer(nn.Module):
                def forward(
                    self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None
                ) -> torch.Tensor:
                    captured_masks.clear()
                    captured_masks.append(src_key_padding_mask)
                    return x + 2.0

            transformer = ShiftTransformer()
            cast(Any, encoder).transformer = transformer
            cast(Any, encoder).input_projection = nn.Identity()
            cast(Any, encoder).dropout_layer = nn.Identity()
            cast(Any, encoder).projection = nn.Identity()

            sequence = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
            output = encoder(sequence)

            assert not captured_masks or captured_masks[0] is None
            expected = (sequence + 2.0).mean(dim=1)
            torch.testing.assert_close(output, expected)
        except NotImplementedError:
            pytest.skip("Transformer SequenceEncoder not implemented yet")

    def test_transformer_encoder_missing_modules_raises(self):
        """Missing transformer components should raise a descriptive error."""
        try:
            encoder = SequenceEncoder(
                input_dim=4,
                hidden_dim=4,
                output_dim=4,
                num_layers=1,
                encoder_type="transformer",
            )
            cast(Any, encoder).input_projection = None
            cast(Any, encoder).transformer = None

            sequence = torch.randn(2, 3, 4)

            with pytest.raises(RuntimeError, match="Transformer modules not initialized."):
                encoder(sequence)
        except NotImplementedError:
            pytest.skip("Transformer SequenceEncoder not implemented yet")


class TestFrameEncoder:
    """Test FrameEncoder module."""

    @pytest.fixture
    def encoder_params(self):
        return {"frame_dim": 512, "hidden_dim": 256, "output_dim": 128}

    @pytest.fixture
    def frame_data(self):
        batch_size = 4
        num_frames = 30
        frame_dim = 512
        return torch.randn(batch_size, num_frames, frame_dim)

    def test_average_pooling(self, encoder_params, frame_data):
        """Test FrameEncoder with average pooling."""
        try:
            encoder = FrameEncoder(**encoder_params, temporal_pooling="average")
            output = encoder(frame_data)

            batch_size = frame_data.size(0)
            expected_shape = (batch_size, encoder_params["output_dim"])

            assert output.shape == expected_shape, (
                f"Expected shape {expected_shape}, got {output.shape}"
            )
            print("✓ FrameEncoder average pooling test passed")
        except NotImplementedError:
            pytest.skip("FrameEncoder average pooling not implemented yet")

    def test_attention_pooling(self, encoder_params, frame_data):
        """Test FrameEncoder with attention pooling."""
        try:
            encoder = FrameEncoder(**encoder_params, temporal_pooling="attention")
            output = encoder(frame_data)

            batch_size = frame_data.size(0)
            expected_shape = (batch_size, encoder_params["output_dim"])

            assert output.shape == expected_shape, (
                f"Expected shape {expected_shape}, got {output.shape}"
            )
            print("✓ FrameEncoder attention pooling test passed")
        except NotImplementedError:
            pytest.skip("FrameEncoder attention pooling not implemented yet")

    def test_with_mask(self, encoder_params):
        """Test FrameEncoder with variable-length videos."""
        try:
            encoder = FrameEncoder(**encoder_params, temporal_pooling="attention")

            batch_size = 2
            num_frames = 30
            frames = torch.randn(batch_size, num_frames, encoder_params["frame_dim"])
            mask = torch.zeros(batch_size, num_frames)
            mask[0, :20] = 1  # First video has 20 frames
            mask[1, :15] = 1  # Second video has 15 frames

            output = encoder(frames, mask)

            assert output.shape == (batch_size, encoder_params["output_dim"])
            assert not torch.isnan(output).any(), "Output contains NaN with mask"
            print("✓ FrameEncoder mask test passed")
        except NotImplementedError:
            pytest.skip("FrameEncoder mask handling not implemented yet")

    def test_frame_encoder_unknown_pooling_raises(self, encoder_params):
        """Invalid pooling strategy should raise at construction time."""
        with pytest.raises(ValueError, match="Unknown pooling"):
            FrameEncoder(**encoder_params, temporal_pooling="median")

    def test_frame_encoder_invalid_input_dim_raises(self, encoder_params):
        """Forward pass validates dimensionality."""
        encoder = FrameEncoder(**encoder_params, temporal_pooling="average")
        with pytest.raises(ValueError, match="Expected 3D frame tensor"):
            encoder(torch.randn(encoder_params["frame_dim"]))

    def test_frame_encoder_average_pooling_with_mask(self, encoder_params):
        """Average pooling branch should handle masks."""
        encoder = FrameEncoder(**encoder_params, temporal_pooling="average")
        frames = torch.randn(2, 5, encoder_params["frame_dim"])
        mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float32)
        output = encoder(frames, mask)
        assert output.shape == (2, encoder_params["output_dim"])

    def test_frame_encoder_max_pooling_with_mask(self, encoder_params):
        """Max pooling branch should mask inactive frames."""
        encoder = FrameEncoder(**encoder_params, temporal_pooling="max")
        frames = torch.randn(2, 4, encoder_params["frame_dim"])
        mask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 0]], dtype=torch.float32)
        output = encoder(frames, mask)
        assert output.shape == (2, encoder_params["output_dim"])

    def test_frame_encoder_max_pooling_without_mask(self, encoder_params):
        """Max pooling without mask uses simple reduction."""
        encoder = FrameEncoder(**encoder_params, temporal_pooling="max")
        frames = torch.randn(2, 4, encoder_params["frame_dim"])
        output = encoder(frames)
        assert output.shape == (2, encoder_params["output_dim"])

    def test_frame_encoder_runtime_unknown_pooling(self, encoder_params):
        """Changing pooling strategy at runtime should be validated."""
        encoder = FrameEncoder(**encoder_params, temporal_pooling="average")
        cast(Any, encoder).temporal_pooling = "bogus"
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            encoder(torch.randn(2, 3, encoder_params["frame_dim"]))

    def test_attention_pool_requires_layer(self, encoder_params):
        """Calling attention pool without attention layer should raise."""
        encoder = FrameEncoder(**encoder_params, temporal_pooling="average")
        with pytest.raises(RuntimeError, match="Attention layer not initialized"):
            encoder.attention_pool(torch.randn(2, 3, encoder_params["hidden_dim"]))


class TestSimpleMLPEncoder:
    """Test SimpleMLPEncoder module."""

    @pytest.fixture
    def encoder_params(self):
        return {"input_dim": 256, "hidden_dim": 128, "output_dim": 64, "num_layers": 2}

    def test_output_shape(self, encoder_params):
        """Test SimpleMLPEncoder output shape."""
        try:
            encoder = SimpleMLPEncoder(**encoder_params)

            batch_size = 4
            features = torch.randn(batch_size, encoder_params["input_dim"])
            output = encoder(features)

            expected_shape = (batch_size, encoder_params["output_dim"])
            assert output.shape == expected_shape, (
                f"Expected shape {expected_shape}, got {output.shape}"
            )
            print("✓ SimpleMLPEncoder shape test passed")
        except NotImplementedError:
            pytest.skip("SimpleMLPEncoder not implemented yet")

    def test_gradient_flow(self, encoder_params):
        """Test gradient flow through SimpleMLPEncoder."""
        try:
            encoder = SimpleMLPEncoder(**encoder_params)

            features = torch.randn(2, encoder_params["input_dim"], requires_grad=True)
            output = encoder(features)
            loss = output.sum()
            loss.backward()

            assert features.grad is not None, "No gradient for input features"
            has_param_grad = any(p.grad is not None for p in encoder.parameters())
            assert has_param_grad, "No gradients in model parameters"
            print("✓ SimpleMLPEncoder gradient flow test passed")
        except NotImplementedError:
            pytest.skip("SimpleMLPEncoder not implemented yet")

    def test_simple_mlp_invalid_input_dim_raises(self, encoder_params):
        """SimpleMLPEncoder should validate feature dimensionality."""
        encoder = SimpleMLPEncoder(**encoder_params)
        with pytest.raises(ValueError, match="Expected 2D feature tensor"):
            encoder(torch.randn(2, 3, encoder_params["input_dim"]))


class TestEncoderFactory:
    """Test build_encoder factory function."""

    def test_video_encoder(self):
        """Test factory creates correct encoder for video."""
        try:
            encoder = build_encoder(modality="video", input_dim=512, output_dim=128)
            assert encoder is not None
            # Should be FrameEncoder
            print("✓ Video encoder factory test passed")
        except NotImplementedError:
            pytest.skip("Video encoder not implemented yet")

    def test_unknown_modality_uses_mlp(self):
        """Fallback should use SimpleMLPEncoder for unknown modalities."""
        encoder = build_encoder(modality="metadata", input_dim=32, output_dim=16)
        assert isinstance(encoder, SimpleMLPEncoder)

    def test_imu_encoder(self):
        """Test factory creates correct encoder for IMU."""
        try:
            encoder = build_encoder(modality="imu", input_dim=64, output_dim=128)
            assert encoder is not None
            # Should be SequenceEncoder
            print("✓ IMU encoder factory test passed")
        except NotImplementedError:
            pytest.skip("IMU encoder not implemented yet")


def test_encoders_module_entrypoint(capsys):
    """Execute encoders.__main__ block for coverage."""
    runpy.run_module("encoders", run_name="__main__")
    output = capsys.readouterr().out
    assert "Testing encoders" in output


def test_encoders_main_handles_errors(monkeypatch, capsys):
    """Ensure __main__ error branches emit diagnostics."""
    import encoders as enc_module

    def execute_main() -> str:
        lines = Path("src/encoders.py").read_text().splitlines()
        start = next(idx for idx, line in enumerate(lines) if line.startswith("if __name__"))
        block = "\n" * start + "\n".join(lines[start:])
        namespace = dict(enc_module.__dict__)
        namespace["__name__"] = "__main__"
        capsys.readouterr()
        exec(compile(block, "src/encoders.py", "exec"), namespace)
        return capsys.readouterr().out.lower()

    seq_init = enc_module.SequenceEncoder.__init__
    seq_forward = enc_module.SequenceEncoder.forward
    frame_init = enc_module.FrameEncoder.__init__
    frame_forward = enc_module.FrameEncoder.forward
    mlp_init = enc_module.SimpleMLPEncoder.__init__
    mlp_forward = enc_module.SimpleMLPEncoder.forward

    def raise_notimpl(*_args, **_kwargs):
        raise NotImplementedError("stub")

    monkeypatch.setattr(enc_module.SequenceEncoder, "__init__", raise_notimpl, raising=False)
    monkeypatch.setattr(enc_module.FrameEncoder, "__init__", raise_notimpl, raising=False)
    monkeypatch.setattr(enc_module.SimpleMLPEncoder, "__init__", raise_notimpl, raising=False)
    output = execute_main()
    assert output.count("not implemented yet") >= 3

    monkeypatch.setattr(enc_module.SequenceEncoder, "__init__", seq_init, raising=False)
    monkeypatch.setattr(enc_module.FrameEncoder, "__init__", frame_init, raising=False)
    monkeypatch.setattr(enc_module.SimpleMLPEncoder, "__init__", mlp_init, raising=False)

    def raise_runtime(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(enc_module.SequenceEncoder, "forward", raise_runtime, raising=False)
    monkeypatch.setattr(enc_module.FrameEncoder, "forward", raise_runtime, raising=False)
    monkeypatch.setattr(enc_module.SimpleMLPEncoder, "forward", raise_runtime, raising=False)
    output = execute_main()
    assert "encoder error" in output
    assert "frameencoder error" in output
    assert "simplemlpencoder error" in output

    monkeypatch.setattr(enc_module.SequenceEncoder, "forward", seq_forward, raising=False)
    monkeypatch.setattr(enc_module.FrameEncoder, "forward", frame_forward, raising=False)
    monkeypatch.setattr(enc_module.SimpleMLPEncoder, "forward", mlp_forward, raising=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
