# Apple Fruit Quality Classification

This project is an end-to-end deep learning system to classify apple fruit quality into 5 categories (Rotten to Fresh), achieving **96.2% accuracy** on real-world datasets with significant class imbalance.

Fine-tuned **MobileNet V2** with advanced augmentation techniques such as **MixUp**, **color jitter**, and **CLAHE**, enabling better generalization in diverse real-world field conditions.

The model was also optimized for **real-time edge inference** via ONNX conversion and batch-free evaluation.

---

##  Training & Validation Performance

Below is the training and validation loss and accuracy across 40 epochs, showing stable convergence and strong generalization:

![Training and Validation Metrics](/data.png)

- **Left Plot:** Loss curves with moving averages showing stable and decreasing loss for both training and validation.
- **Right Plot:** Accuracy curves stabilizing near 98% training and 96–98% validation accuracy.
- Minimal overfitting was observed, and training was consistent due to effective data augmentation and regularization.

---

##  Project Structure

- `app/` - Application code for inference
- `models/` - Trained ONNX model(s)
- `training/` - Jupyter notebooks and scripts for model training
- `requirements.txt` - Python dependencies

---

##  Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/apple-fruit-quality.git
   cd apple-fruit-quality
