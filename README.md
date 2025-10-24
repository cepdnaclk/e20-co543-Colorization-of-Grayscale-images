# Colourization of Grayscale Images using Deep Learning

This repository presents a deep learning-based approach for automatic colorization of grayscale images using Convolutional Neural Networks (CNNs) and U-Net architectures. The project was conducted as part of an academic research study at the Department of Computer Engineering, University of Peradeniya.

---

## Overview

Image colorization is a long-standing problem in computer vision. Traditional colorization relies on manual editing, which is both time-consuming and subjective. Deep learning offers a data-driven approach to infer realistic colors automatically.

This work focuses on implementing and evaluating CNN-based and U-Net-based colorization models. The models were trained in the CIE LAB color space, where the network learns to predict the A and B chrominance channels from the grayscale L channel. The study also explores the effect of different datasets and analyzes computational limitations encountered during large-scale training.

---

## Objectives

* Develop a baseline CNN model for grayscale image colorization.
* Implement a U-Net model to improve spatial and contextual feature extraction.
* Evaluate model performance on CIFAR-10, ImageNet, and Places365 datasets.
* Compare results using quantitative metrics (MSE, PSNR, SSIM) and qualitative evaluation.
* Identify key challenges and propose directions for improvement using GANs and self-supervised methods.

---

## Methodology

1. **Color Space Conversion**
   Images were converted from RGB to CIE LAB color space. The L (lightness) channel was used as input, while the model predicted the A and B (chrominance) channels.

2. **Model Architectures**

   * **CNN:** An 8-layer convolutional network with ReLU and tanh activations was used as the baseline.
   * **U-Net:** An encoder-decoder architecture with skip connections was implemented to preserve spatial details.

3. **Training Setup**

   * Optimizer: Adam
   * Loss Function: Mean Squared Error (MSE)
   * Datasets: CIFAR-10, subsets of ImageNet and Places365
   * Evaluation Metrics: MSE, PSNR, SSIM
   * Training Platform: Google Colab (GPU environment)

---

## Results

| Metric    | Score    |
| --------- | -------- |
| MSE       | 0.0092   |
| PSNR      | 20.41 dB |
| Accuracy  | 0.5791   |
| Precision | 0.4414   |
| Recall    | 0.1593   |
| F1-Score  | 0.1620   |

The models successfully learned basic color mapping patterns but struggled to capture global semantics. Computational limitations restricted large-scale training, especially with ImageNet and Places365 datasets.

---

## Technologies Used

* Python 3.x
* PyTorch
* NumPy, Matplotlib, OpenCV
* Google Colab

---

## Key Findings

* LAB color space improves perceptual quality and model stability compared to RGB.
* CNNs perform well on small datasets but lack global contextual understanding.
* U-Net improves spatial consistency but remains sensitive to dataset diversity.
* Resource limitations (memory and runtime) significantly affect large-scale training feasibility.

---

## Future Work

* Introduce Generative Adversarial Networks (GANs) for perceptually realistic colorization.
* Explore self-supervised and contrastive learning for unlabeled data.
* Integrate pretrained encoders (e.g., ResNet) to improve semantic understanding.
* Optimize data pipelines and hardware utilization for large-scale datasets.

---

## Team Members and Contributions

This project was carried out as a **group research project** by undergraduate students of the **Department of Computer Engineering, University of Peradeniya**, under the supervision of the CO543 course instructors.

| Member | Student ID | Contribution |
|---------|-------------|--------------|
| **D.M.T. Dilshan** | E/20/069 | Model architecture, MLOps integration, documentation |
| **R.V.C. Rathnaweera** | E/20/328 | Data preprocessing, model fine-tuning |
| **K.N.P. Karunarathne** | E/20/189 | Literature review, model evaluation |
| **W.M.N. Dilshan** | E/20/455 | Dataset preparation, report writing |

---


## Acknowledgments

This project was carried out under the guidance of the Department of Computer Engineering, University of Peradeniya. The authors acknowledge the use of Google Colab for experimental work and thank the open-source PyTorch community for the resources used throughout this study.

---

## Citation

If you refer to this work, please cite:

Dishan D.M.T., Karunarathne K.N.P., Rathnaweera R.V.C., and Dishan W.M.N.,
"Colourisation of Grayscale Images," Department of Computer Engineering,
University of Peradeniya, 2025.

---

## Example Outputs

(Add sample results comparing grayscale and colorized outputs once available.)

---

## Repository Structure

```
├── Backend/
├── Frontend/
├── CO543_MiniProjectProposal_Group16/
├── ImageColorization.ipynb
├── LICENSE
├── Project_Report_Paper_.pdf
└── README.md
```

---

## Keywords

Image Colorization, Deep Learning, CNN, U-Net, Computer Vision, PyTorch, LAB Color Space, Image Restoration
