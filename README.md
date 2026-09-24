
# Deep Image Prior for Image Denoising

A PyTorch implementation of **Deep Image Prior (DIP)** for image denoising without requiring a pretrained model or clean training dataset.

The project explores how the structure of a randomly initialized convolutional neural network can act as an implicit prior for natural images and reconstruct a clean image from a noisy observation.

---

## 📌 Overview

Traditional image denoising methods generally rely on either:

* Large datasets of clean/noisy image pairs
* Pretrained neural networks
* Explicit assumptions about image statistics

**Deep Image Prior** takes a different approach.

Instead of training a model on a dataset, DIP optimizes a randomly initialized neural network directly for the **single noisy image**.

The network gradually learns the underlying image structure while high-frequency noise tends to be fitted later. This makes **early stopping** an important part of the method.

### Core idea

Given a noisy image:

```text
        Noisy Image
             │
             ▼
    ┌─────────────────┐
    │ Random Network  │
    │      θ          │
    └─────────────────┘
             │
             ▼
      Reconstructed
         Image
             │
             ▼
     Compare with Noisy
          Image
             │
             ▼
       Update θ
```

The optimization can be represented as:

$$
\theta^* = \arg\min_\theta
\left\|f_\theta(z)-y\right\|^2
$$

where:

* \(y\) = noisy input image
* \(z\) = fixed random noise input
* \(f_\theta\) = randomly initialized neural network
* \(\theta\) = network parameters

The final output is:

$$
x^* = f_{\theta^*}(z)
$$

---

## ✨ Features

* Dataset-free image denoising
* PyTorch implementation
* Randomly initialized CNN architecture
* Supports different noise levels
* PSNR evaluation
* SSIM evaluation
* Training-loss monitoring
* Output image generation
* Experiment tracking
* PSNR visualization
* GPU acceleration when available

---

## 🧠 Why Does DIP Work?

One of the interesting properties of DIP is that the neural network architecture itself provides an **implicit image prior**.

During optimization, the network tends to learn:

1. Large-scale image structures
2. Shapes and edges
3. Textures and finer details
4. High-frequency noise

Therefore, the output can initially become cleaner even though the network is trained **only against the noisy image**.

This leads to an important observation:

> Training for too long can cause the network to memorize the noise.

Therefore, **early stopping** is critical.

---

## 🏗️ Architecture

The implementation uses a convolutional encoder-decoder style network.


<img width="6229" height="5136" alt="diagram" src="https://github.com/user-attachments/assets/546192cf-5229-4178-b6db-02ff89820e0c" />

The network parameters are optimized while the input noise tensor remains fixed.

---

## 🔬 Experimental Setup

Experiments can be performed by varying the amount of noise added to an image.

Example noise levels:

```text
σ = 10
σ = 25
σ = 50
σ = 75
```

For every noise level, the model is optimized independently.

The resulting image quality can then be evaluated using:

### PSNR

Peak Signal-to-Noise Ratio measures reconstruction quality.

Higher PSNR generally indicates better similarity to the reference image.

$$
PSNR = 10\log_{10}\left(\frac{MAX_I^2}{MSE}\right)
$$

### SSIM

Structural Similarity Index measures similarity in terms of:

* Luminance
* Contrast
* Structural information

SSIM values generally range from:

```text
0 → Poor structural similarity
1 → Identical images
```

---

## 📊 Results

The experiments track image quality throughout training.

Typical behavior:

```text
PSNR
 │
 │          /───────\
 │        /           \
 │      /
 │    /
 │  /
 │ /
 └─────────────────────────► Iterations
          ↑
      Best Point
```

Initially, the network learns useful image structure and PSNR improves.

After a certain point, the network begins fitting the noise, causing reconstruction quality to deteriorate.

This demonstrates why selecting an appropriate stopping point is important.

---

## 📁 Project Structure

```text
deep-image-prior/
│
├── experiments/
│   ├── results/
│   ├── plots/
│   └── outputs/
│
├── images/
│   ├── input/
│   └── output/
│
├── models/
│   └── dip_network.py
│
├── utils/
│   ├── image_utils.py
│   └── metrics.py
│
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

> The exact structure may vary depending on the implementation.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd deep-image-prior
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

Place the noisy image in the appropriate input directory.

Then run:

```bash
python train.py
```

The model will:

1. Load the noisy image
2. Generate the fixed random input
3. Initialize the CNN
4. Optimize the network parameters
5. Generate reconstructed images
6. Calculate evaluation metrics
7. Save the best reconstruction
8. Generate experiment plots

---

## 🖥️ Hardware

The implementation supports:

* CPU
* NVIDIA CUDA GPUs
* Apple Silicon through MPS

For Apple Silicon:

```python
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
```

For CUDA:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

---

## 📈 Experiment Tracking

Each experiment can record:

```text
Noise Level
Iterations
Loss
PSNR
SSIM
Best Iteration
Training Time
```

Example:

| Noise σ | Best Iteration | PSNR | SSIM |
| ------: | -------------: | ---: | ---: |
|      10 |              — |    — |    — |
|      25 |              — |    — |    — |
|      50 |              — |    — |    — |
|      75 |              — |    — |    — |

Replace the values with the results generated by your experiments.

---

## 🔍 Key Observations

The experiments demonstrate several important characteristics of Deep Image Prior:

* A randomly initialized CNN can reconstruct meaningful image structure.
* No external training dataset is required.
* Network architecture acts as an implicit prior.
* Lower-frequency structures are generally reconstructed before high-frequency noise.
* Excessive optimization can result in noise fitting.
* Early stopping significantly affects reconstruction quality.
* PSNR and SSIM can be used to identify the best reconstruction during experimentation.

---

## 🧪 Future Improvements

Potential improvements include:

* Automatic early-stopping strategies
* More advanced DIP architectures
* Comparison with classical denoising methods
* Comparison with supervised deep-learning denoisers
* Additional image-quality metrics
* Hyperparameter optimization
* Multi-scale architectures
* Blind noise-level estimation
* Real-world noise experiments
* Improved MPS/CUDA performance
* Extensive benchmark evaluation

---

## 📚 References

The project is based on the research paper:

**Deep Image Prior**

Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky

The key idea is that the structure of a randomly initialized neural network can serve as a powerful prior for solving inverse problems such as:

* Image denoising
* Image super-resolution
* Image inpainting

---

## 🛠️ Technologies

* **Python**
* **PyTorch**
* **NumPy**
* **OpenCV / PIL**
* **Matplotlib**
* **scikit-image**

---

## 👨‍💻 Author

**Punith N S**

Computer Science & Engineering
Siddaganga Institute of Technology

---

## ⭐ Acknowledgements

This project is inspired by the original **Deep Image Prior** research by Ulyanov, Vedaldi, and Lempitsky.

If you find this implementation useful, consider giving the repository a ⭐.
