# Tropical-Cyclone-ConvLSTM-Model

This repository implements a ConvLSTM-based deep learning model for tropical cyclone (typhoon/hurricane) track prediction. The model integrates adversarial training ideas from GANs into a ConvLSTM backbone, introducing an adversarial loss term to improve prediction accuracy and stability.

Main contents
- Model implementation: `Model.py` (contains model definitions, training and inference routines; adaptable to different data formats)
- Example data: `Data_example/example.npz` (demonstrates the expected data organization)
- Output example: `Output_example/example.jpg` (example of a predicted track visualization)
- Model architecture: `Model Architecture/Model Architecture.jpg` (diagram of the model)

Data format
To run the examples directly, sample data is provided. The model was developed using CMA best-track data combined with ERA5 reanalysis fields; after extracting ERA5 fields along cyclone coordinates, the data should be packaged into an `.npz` file.

Dependencies
- Python 3.8+
- PyTorch (recommended 1.8+)
- NumPy
- scikit-learn
- Matplotlib (for visualization)

Recommended conda environment setup:

```bash
conda create -n convlstm python=3.8 -y
conda activate convlstm
pip install torch numpy scikit-learn matplotlib
```

(Select the appropriate `torch` installation for your CUDA/CPU environment; see https://pytorch.org)

Quick start (example)
1. Prepare your data and ensure it follows the example data format.
2. Adjust model settings in `Model.py` as needed.
3. Run training (the default settings in `Model.py` are example-level; read inline comments and adapt hyperparameters before large runs):

```bash
# Run from the project root
python Model.py
```

Training notes and recommendations
- Learning rate: common starting values are 1e-3 or 1e-4; consider using a scheduler.
- Convolution kernel sizes: avoid excessively large kernels as they may hurt performance.
- Batch size: limited by GPU memory; reduce `batch_size` if you encounter out-of-memory errors.

We recommend first validating the pipeline with a small dataset and a smaller model to ensure correct data flow and reasonable loss behavior, then scale up experiments.

Outputs and visualization
- Model outputs are typically predicted fields or track coordinates; `Model.py` includes basic plotting utilities (requires `matplotlib`).
- Save checkpoints including model weights and optimizer state to allow resuming training and reproducible evaluation.

Example data note
`Data_example/example.npz` is a minimal example showing the expected data structure. Do not use it for performance evaluation—only as a format reference.

