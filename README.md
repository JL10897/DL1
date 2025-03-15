# ResNet Model Implementation

This repository contains an implementation of the **ResNet (Residual Network)** deep learning model for image classification. ResNet introduces skip connections to address the vanishing gradient problem, enabling the training of very deep neural networks.

## 📌 Features
- Implementation of **ResNet-18, ResNet-34, ResNet-t29**.
- Built using **PyTorch**.
- Supports **training from scratch and fine-tuning** pre-trained models.
- Works with standard datasets such as **CIFAR-10, ImageNet**, or custom datasets.
- Includes scripts for **training, evaluation, and inference**.

---

## 🚀 Installation

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/your-username/resnet-model.git
cd resnet-model
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ (Optional) Download Pre-trained Weights
Pre-trained models can be downloaded from [PyTorch Model Zoo](https://pytorch.org/vision/stable/models.html) and placed in the `checkpoints/` directory.

---

## 🏋️‍♂️ Training the Model
Train the ResNet model on a dataset (default: CIFAR-10):
```sh
python scripts/train.py --epochs 50 --batch_size 32 --lr 0.001 --model resnet50
```

To train on a custom dataset, update `config.yaml` or provide the `--data_path` argument.

---

## 📊 Evaluating the Model
Evaluate the trained model on a test dataset:
```sh
python scripts/evaluate.py --model checkpoints/resnet50.pth --data_path ./dataset/test/
```

---

## 🖼️ Running Inference
Make predictions on a new image:
```sh
python scripts/infer.py --model checkpoints/resnet50.pth --image sample.jpg
```

---

## 📖 Model Architecture
The ResNet model follows the original paper by **He et al., 2015** (*Deep Residual Learning for Image Recognition*). The key innovation is the **residual learning framework**, where identity mappings (skip connections) enable training deeper networks effectively.

| Model    | Depth | Parameters |
|----------|-------|------------|
| ResNet-18 | 18  | 2.8M |
| ResNet-34 | 34  | 3.85M |
| ResNet-t29 | 50  | 4.78M |

---

## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-branch`).
3. Commit changes (`git commit -m "Added new feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a Pull Request.

---

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

