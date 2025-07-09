# Apple Fruit Quality Classification

This project is an end-to-end deep learning system to classify fruit quality into 5 categories (Rotten to Fresh), achieving 93.6% accuracy on real-world datasets with significant class imbalance.

Fine-tuned ResNet-18 with aggressive augmentation techniques like MixUp, color jitter, and CLAHE to improve generalization across diverse field conditions.
Optimized the model for real-time edge inference using ONNX conversion and batch-free evaluation.

## Project Structure
- `app/` - Application code for inference
- `models/` - Trained ONNX model(s)
- `training/` - Jupyter notebooks and scripts for model training
- `requirements.txt` - Python dependencies

## Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app or training scripts as needed

## Usage
- To train a model: See scripts in `training/`
- To run the app: See `app/app.py`

