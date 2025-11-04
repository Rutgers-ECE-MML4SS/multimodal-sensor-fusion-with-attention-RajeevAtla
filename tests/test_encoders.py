"""
Unit tests for encoder modules.

Tests basic functionality and interface compliance.
"""

import pytest
import torch
import sys
from pathlib import Path

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

    def test_imu_encoder(self):
        """Test factory creates correct encoder for IMU."""
        try:
            encoder = build_encoder(modality="imu", input_dim=64, output_dim=128)
            assert encoder is not None
            # Should be SequenceEncoder
            print("✓ IMU encoder factory test passed")
        except NotImplementedError:
            pytest.skip("IMU encoder not implemented yet")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
