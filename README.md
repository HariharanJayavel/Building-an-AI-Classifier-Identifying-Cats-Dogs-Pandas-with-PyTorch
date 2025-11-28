# Building an AI Classifier: Identifying Cats, Dogs & Pandas with PyTorch

This project demonstrates how to build an image classification model using **transfer learning** in **PyTorch** to classify images into **three categories: Cat, Dog, and Panda**.

---

## 1. Environment Setup

### Verify PyTorch & CUDA

```python
import torch

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
```

### If running locally
- Install Python 3.9+
- Ensure NVIDIA GPU available
- Install CUDA Toolkit + drivers

### If running on Kaggle
- Go to Settings → Accelerator → GPU
- Install requirements:

```bash
pip install torch torchvision torchaudio --upgrade
```

---

## 2. Data Preparation

### Download Dataset (Kaggle)

```
kaggle datasets download -d gpiosenka/cats-dogs-pandas-images -p ./data --unzip
```

### Folder Structure

```
data/
  train/
    cat/
    dog/
    panda/
  test/
    cat/
    dog/
    panda/
```

### Transforms (PyTorch ImageFolder)
- Resize: 224×224  
- Normalize: ImageNet mean & std  
- Augmentations: flip, rotation, crop  

---

## 3. Model Design (Transfer Learning)

### Select Pre-Trained Model
ResNet18, VGG16, EfficientNet, MobileNetV2

### Modify Classifier
```
FC → 256 neurons → ReLU → Dropout(0.5)
Output → 3 classes (cat, dog, panda)
```

### Move Model to GPU

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

---

## 4. Training

- Loss: CrossEntropyLoss  
- Optimizer: Adam (lr=0.001)  
- Epochs: 10–15  
- Save best model  

---

## 5. Evaluation

Report:
- Test Loss  
- Test Accuracy  
- Confusion Matrix  
- Sample Predictions  

---

## 6. BONUS (Optional)

### Predict custom image
```
predict_image("path/to/image.jpg")
```

### Deployment
- Streamlit App  
- Flask API  

---

## 7. requirements.txt

```
torch
torchvision
torchaudio
matplotlib
numpy
pandas
scikit-learn
kaggle
tqdm
Pillow
ipykernel
jupyter
notebook
seaborn
```

---

## 8. Files to include in GitHub

```
notebooks/cat_dog_panda_transfer_learning.ipynb
requirements.txt
README.md
```
