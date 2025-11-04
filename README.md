[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/n_Dh2nMC)
# A2: Multimodal Sensor Fusion with Attention

**ECE 532 - Fall 2025**  
**Release: October 25, 2025 • Due: November 10, 2025 23:59 ET**

This repository contains the starter code for Assignment 2, focusing on attention-based multimodal sensor fusion for heterogeneous sensor streams.

## 🚀 Quick Start

### Local Environment Setup

1. **Clone repository and set up environment:**
   ```bash
   cd a2-<your-netid>
   conda env create -f environment.yml -n a2
   conda activate a2
   ```

2. **Run quick tests:**
   ```bash
   pytest -q
   ```

## 📁 Repository Structure

```
a2/
├── src/
│   ├── fusion.py           # TODO: Implement fusion architectures
│   ├── attention.py        # TODO: Implement attention mechanisms
│   ├── encoders.py         # TODO: Implement encoders
│   ├── uncertainty.py      # TODO: Implement uncertainty quantification
│   ├── train.py            # Training pipeline (mostly complete)
│   ├── eval.py             # Evaluation script (mostly complete)
│   ├── analysis.py         # Generate plots (mostly complete)
│   └── data.py             # Dataset loading (complete)
├── config/
│   ├── base.yaml           # Base configuration
│   ├── fusion_strategies.yaml  # Fusion strategy configs
│   └── datasets.yaml       # Dataset-specific configs
├── experiments/            # Your experiment results (JSON)
├── analysis/              # Your generated plots (PNG)
├── runs/                  # Model checkpoints
├── tests/                 # Unit tests for your implementations
├── report.pdf            # Your report (≤4 pages)
└── README.md
```

## 🎯 Assignment Requirements

### Core Implementations (Required)

You need to implement the following modules:

**1. Fusion Architectures (`src/fusion.py`)** - 35 points
- `EarlyFusion`: Concatenate features before processing
- `LateFusion`: Independent processing, combine predictions
- `HybridFusion`: Cross-modal attention + learned weighting (main focus!)

**2. Attention Mechanisms (`src/attention.py`)** - 15 points
- `CrossModalAttention`: Attention between different modalities
- Visualize attention weights

**3. Encoders (`src/encoders.py`)** - See note below
- `SequenceEncoder`: For time-series data (IMU, audio)
- `FrameEncoder`: For frame-based data (video)
- `SimpleMLPEncoder`: For pre-extracted features

**Note:** Pick ONE encoder type per modality that matches your dataset. You don't need to implement all encoder variants.

**4. Uncertainty Quantification (`src/uncertainty.py`)** - 20 points
- Expected Calibration Error (ECE)
- Reliability diagrams
- Target ECE ≈ 0.1

**5. Missing Modality Handling** - 25 points
- Test with all possible modality subsets
- Graceful performance degradation

**6. Report & Analysis** - 5 points
- ≤4 pages technical report
- All required plots and tables

See `A2.pdf` for full details and rubric.

## 📊 Datasets

You must use one of the following real datasets:

**PAMAP2 (Recommended):**
- Download: https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring
- Smallest dataset (~500MB)
- CPU-friendly (no video processing)
- 3 IMU sensors + heart rate
- See A2.pdf for preprocessing instructions

**MHAD:**
- Download: http://tele-immersion.citris-uc.org/berkeley_mhad
- Video + IMU data
- Multiple sampling rates
- See A2.pdf for preprocessing instructions

**MPI Cooking:**
- Download: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/human-activity-recognition/mpii-cooking-2-dataset
- Audio-visual cooking actions
- See A2.pdf for preprocessing instructions

## 💻 Development Workflow

### Step 1: Implement Core Modules (Week 1)

Start with the simplest versions:

1. **Implement `EarlyFusion` first**:
   ```bash
   # Edit src/fusion.py
   # Test with:
   python -m pytest tests/test_fusion.py::TestFusionInterfaces::test_early_fusion_shape -v
   ```

2. **Implement `SimpleMLPEncoder`** (easiest encoder):
   ```bash
   # Edit src/encoders.py
   # Test with:
   python -m pytest tests/test_encoders.py::TestSimpleMLPEncoder -v
   ```

3. **Download and prepare your dataset** (see Datasets section above)

### Step 2: Add Attention and Hybrid Fusion (Week 2)

1. **Implement `CrossModalAttention`**:
   ```bash
   # Edit src/attention.py
   # Test with:
   python -m pytest tests/test_attention.py::TestCrossModalAttention -v
   ```

2. **Implement `HybridFusion`** using your attention module

3. **Compare fusion strategies**:
   ```bash
   # Train all three fusion types
   for fusion in early late hybrid; do
       python src/train.py model.fusion_type=$fusion
   done
   ```

### Step 3: Experiments and Analysis (Week 3)

1. **Run missing modality tests**:
   ```bash
   python src/eval.py --checkpoint runs/best.ckpt --missing_modality_test
   ```

2. **Generate plots**:
   ```bash
   python src/analysis.py --experiment_dir experiments/ --output_dir analysis/
   ```

3. **Write report** with your results and analysis

## 🧪 Testing Your Implementation

```bash
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_fusion.py -v

# Run specific test
pytest tests/test_fusion.py::TestFusionInterfaces::test_early_fusion_shape -v
```

Tests check:
- Correct input/output shapes
- Gradient flow
- Missing modality handling
- No NaN values in outputs

## 📝 Required Outputs

For grading, ensure these files exist:

### Experiment Results (JSON)
- `experiments/fusion_comparison.json`
- `experiments/missing_modality.json`
- `experiments/uncertainty.json`

### Analysis Plots (PNG)
- `analysis/fusion_comparison.png`
- `analysis/missing_modality.png`
- `analysis/attention_viz.png`
- `analysis/calibration.png`

### Report
- `report.pdf` (≤4 pages in root directory)

## 🎓 Expected Performance

### On PAMAP2 (recommended dataset)
- Single modality: ~65-70%
- Early fusion: ~75-80%
- Hybrid fusion: ~80-85%

Your results may vary based on hyperparameters and random seed.

## 🔧 Troubleshooting

### Common Issues

**Import errors:**
```bash
# Make sure you're in the right directory and environment
conda activate a2
cd a2-<netid>
```

**Out of memory:**
```bash
# Reduce batch size in config/base.yaml
dataset:
  batch_size: 16  # or even 8
```

**Tests fail with NotImplementedError:**
- This is expected! Implement the functions marked with TODO
- Tests will pass as you implement each module

## 🆘 Getting Help

- **Piazza:** Tag questions with `#a2`
- **Office Hours:** Tuesday & Friday 2-3 PM (SEC-210)
- **Assignment PDF:** See `A2.pdf` for full requirements

## 🏁 Submission

1. **Complete all implementations** and verify tests pass

2. **Generate all required outputs:**
   ```bash
   # Train models
   # Run evaluations
   # Generate plots
   ```

3. **Write report** (≤4 pages)

4. **Tag your submission:**
   ```bash
   git add .
   git commit -m "Final submission for A2"
   git tag submission
   git push origin main --tags
   ```

5. **Submit on Canvas:**
   - Upload `report.pdf`
   - Submit repository link

**Deadline:** November 10, 2025, 23:59 ET

Late submissions: -10% per day (72h max)

---

**Good luck! Focus on understanding fusion concepts and building robust systems. 🎯**

