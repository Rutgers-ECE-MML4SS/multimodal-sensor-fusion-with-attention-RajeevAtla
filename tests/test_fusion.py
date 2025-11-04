"""
Unit tests for fusion architectures.

Tests basic functionality and interface compliance.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fusion import EarlyFusion, LateFusion, HybridFusion, build_fusion_model


class TestFusionIntegration:
    """Integration-style tests covering missing-modality fallbacks and adaptive weights."""

    def test_late_fusion_missing_modality_fallback(self):
        """LateFusion should fallback to uniform weights when modalities are missing."""
        torch.manual_seed(0)
        modality_dims = {"video": 4, "imu": 4}
        model = LateFusion(modality_dims, num_classes=3, hidden_dim=8, dropout=0.0)
        model.eval()

        features = {
            "video": torch.randn(2, 4),
            "imu": torch.randn(2, 4),
        }
        mask = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)

        fused_logits, per_mod_logits = model(features, mask)

        assert torch.allclose(
            fused_logits[0], per_mod_logits["video"][0], atol=1e-6
        ), "Available modality should dominate fused output."

        expected_uniform = (
            per_mod_logits["video"][1] + per_mod_logits["imu"][1]
        ) / 2.0
        assert torch.allclose(
            fused_logits[1], expected_uniform, atol=1e-6
        ), "Fallback should average logits when all modalities missing."

    def test_hybrid_fusion_adaptive_weights(self):
        """HybridFusion adaptive weights should respect modality availability."""
        torch.manual_seed(0)
        modality_dims = {"video": 4, "imu": 4}
        model = HybridFusion(
            modality_dims, num_classes=3, hidden_dim=8, num_heads=1, dropout=0.0
        )
        model.eval()

        features = {
            "video": torch.randn(3, 4),
            "imu": torch.randn(3, 4),
        }
        mask = torch.tensor(
            [[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]], dtype=torch.float32
        )

        logits, attention_info = model(
            features, mask, return_attention=True
        )
        fusion_weights = attention_info["fusion_weights"]

        assert fusion_weights.shape == mask.shape
        assert torch.allclose(
            fusion_weights[0].sum(), torch.tensor(1.0), atol=1e-6
        ), "Weights should remain normalized when all modalities present."
        assert torch.allclose(
            fusion_weights[1], torch.tensor([1.0, 0.0]), atol=1e-6
        ), "Missing modality should receive zero weight."
        assert torch.allclose(
            fusion_weights[2], torch.full((2,), 0.5), atol=1e-6
        ), "All-missing case should fallback to uniform distribution."
        assert not torch.isnan(logits).any(), "Logits should stay finite."


class TestFusionInterfaces:
    """Test that fusion models follow expected interfaces."""

    @pytest.fixture
    def modality_dims(self):
        return {"video": 512, "imu": 64}

    @pytest.fixture
    def num_classes(self):
        return 11

    @pytest.fixture
    def batch_size(self):
        return 4

    @pytest.fixture
    def sample_features(self, batch_size):
        return {
            "video": torch.randn(batch_size, 512),
            "imu": torch.randn(batch_size, 64),
        }

    @pytest.fixture
    def sample_mask(self, batch_size):
        # Different availability patterns
        return torch.tensor(
            [
                [1, 1],  # Both available
                [1, 0],  # Only video
                [0, 1],  # Only IMU
                [1, 1],  # Both available
            ],
            dtype=torch.float,
        )

    def test_early_fusion_shape(
        self, modality_dims, num_classes, sample_features, sample_mask
    ):
        """Test EarlyFusion output shape."""
        try:
            model = EarlyFusion(modality_dims, num_classes=num_classes)
            logits = model(sample_features, sample_mask)

            assert logits.shape == (len(sample_mask), num_classes), (
                f"Expected shape ({len(sample_mask)}, {num_classes}), got {logits.shape}"
            )
            print("✓ EarlyFusion shape test passed")
        except NotImplementedError:
            pytest.skip("EarlyFusion not implemented yet")

    def test_late_fusion_shape(
        self, modality_dims, num_classes, sample_features, sample_mask
    ):
        """Test LateFusion output shape."""
        try:
            model = LateFusion(modality_dims, num_classes=num_classes)
            output = model(sample_features, sample_mask)

            # Late fusion should return tuple (fused_logits, per_modality_logits)
            if isinstance(output, tuple):
                logits, per_mod_logits = output
                assert logits.shape == (len(sample_mask), num_classes)
                assert isinstance(per_mod_logits, dict)
                print("✓ LateFusion shape test passed")
            else:
                logits = output
                assert logits.shape == (len(sample_mask), num_classes)
                print("✓ LateFusion shape test passed (single output)")
        except NotImplementedError:
            pytest.skip("LateFusion not implemented yet")

    def test_hybrid_fusion_shape(
        self, modality_dims, num_classes, sample_features, sample_mask
    ):
        """Test HybridFusion output shape."""
        try:
            model = HybridFusion(modality_dims, num_classes=num_classes, num_heads=4)
            output = model(sample_features, sample_mask, return_attention=False)

            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            assert logits.shape == (len(sample_mask), num_classes), (
                f"Expected shape ({len(sample_mask)}, {num_classes}), got {logits.shape}"
            )
            print("✓ HybridFusion shape test passed")
        except NotImplementedError:
            pytest.skip("HybridFusion not implemented yet")

    def test_hybrid_fusion_attention_output(
        self, modality_dims, num_classes, sample_features, sample_mask
    ):
        """Test HybridFusion returns attention weights when requested."""
        try:
            model = HybridFusion(modality_dims, num_classes=num_classes, num_heads=4)
            logits, attention_info = model(
                sample_features, sample_mask, return_attention=True
            )

            assert logits.shape == (len(sample_mask), num_classes)
            assert attention_info is not None, (
                "Should return attention info when requested"
            )
            print("✓ HybridFusion attention output test passed")
        except NotImplementedError:
            pytest.skip("HybridFusion not implemented yet")

    def test_factory_function(self, modality_dims, num_classes):
        """Test build_fusion_model factory function."""
        for fusion_type in ["early", "late", "hybrid"]:
            try:
                model = build_fusion_model(
                    fusion_type=fusion_type,
                    modality_dims=modality_dims,
                    num_classes=num_classes,
                )
                assert model is not None
                print(f"✓ Factory function works for {fusion_type}")
            except NotImplementedError:
                pytest.skip(f"{fusion_type} fusion not implemented yet")


class TestMissingModalityHandling:
    """Test that models handle missing modalities gracefully."""

    @pytest.fixture
    def model_and_data(self):
        modality_dims = {"video": 512, "imu": 64}
        num_classes = 11
        batch_size = 2

        features = {
            "video": torch.randn(batch_size, 512),
            "imu": torch.randn(batch_size, 64),
        }

        return modality_dims, num_classes, features

    def test_all_modalities_available(self, model_and_data):
        """Test with all modalities available."""
        modality_dims, num_classes, features = model_and_data
        mask = torch.ones(2, 2)

        try:
            model = EarlyFusion(modality_dims, num_classes=num_classes)
            logits = model(features, mask)
            assert not torch.isnan(logits).any(), "Output contains NaN"
            print("✓ All modalities available test passed")
        except NotImplementedError:
            pytest.skip("EarlyFusion not implemented yet")

    def test_one_modality_missing(self, model_and_data):
        """Test with one modality missing."""
        modality_dims, num_classes, features = model_and_data
        mask = torch.tensor([[1, 0], [1, 0]], dtype=torch.float)  # IMU missing

        try:
            model = EarlyFusion(modality_dims, num_classes=num_classes)
            logits = model(features, mask)
            assert not torch.isnan(logits).any(), (
                "Output contains NaN with missing modality"
            )
            print("✓ One modality missing test passed")
        except NotImplementedError:
            pytest.skip("EarlyFusion not implemented yet")


class TestGradientFlow:
    """Test that gradients flow through the models."""

    def test_early_fusion_gradients(self):
        """Test gradient flow in EarlyFusion."""
        try:
            modality_dims = {"video": 512, "imu": 64}
            model = EarlyFusion(modality_dims, num_classes=5)

            features = {
                "video": torch.randn(2, 512, requires_grad=True),
                "imu": torch.randn(2, 64, requires_grad=True),
            }
            mask = torch.ones(2, 2)

            logits = model(features, mask)
            loss = logits.sum()
            loss.backward()

            # Check that model parameters have gradients
            has_grad = any(p.grad is not None for p in model.parameters())
            assert has_grad, "No gradients in model parameters"
            print("✓ EarlyFusion gradient flow test passed")
        except NotImplementedError:
            pytest.skip("EarlyFusion not implemented yet")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
