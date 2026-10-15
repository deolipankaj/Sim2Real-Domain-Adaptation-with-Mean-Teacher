# Paper (TCCML, NeurIPS 2026)
"**Bridging Annotation Granularity in Synthetic-to-Real Forest Tree Segmentation: The MGTD Dataset**" accepted at TCCML@NeurIPS 2026

# Sim2Real Domain Adaptation with Mean Teacher

This project addresses the domain gap between **simulated** and **real forest environments** for **instance segmentation** using a **Mean Teacher framework with knowledge distillation**.

The method leverages **simulated forest data as the source domain** and **real forest data as the target domain**, enabling the model to **recover the domain gap for instance segmentation tasks** by transferring knowledge learned in simulation to real-world forest imagery.

The framework is inspired by the **Mean Teacher approach by Tarvainen & Valpola** (https://github.com/CuriousAI/mean-teacher).


## Motivation

Training perception models for forest environments is difficult because:

1. Collecting large labeled real-world datasets is expensive.
2. Models trained in simulation often fail in real environments due to domain shifts.
3. This work explores **Sim2Real domain adaptation**, where knowledge learned from simulated data is transferred to real forest imagery.


## Method Overview

The proposed framework consists of two models:

### 1. MMT Teacher
- Trained on **simulated forest data**
- Provides reliable supervision signals

### 2. Mean Teacher Student
- Adapted to **real forest data**
- Learns through **consistency training and pseudo-label supervision**

During training, the **teacher network provides distillation signals** to guide the student during adaptation to the real domain.



## Adaptation of Mean Teacher for Instance Segmentation

The original **Mean Teacher framework** is extended for **instance segmentation** with the following modifications:

### 1. Segmentation Architecture
Replacing the classification backbone with a **Mask R-CNN architecture**.

### 2. Object-level Training
Training on **bounding box crops** instead of full-scene images.

### 3. Asymmetric Data Augmentation
- **Teacher:** Weak augmentation  
- **Student:** Strong augmentation  

### 4. Segmentation Loss Functions
- **Supervised Loss:** Dice Loss + Binary Cross Entropy (BCE) with logits  
- **Consistency Loss:** Pixel-wise Mean Squared Error (MSE) between teacher and student predictions  

### 5. Pseudo-label Filtering
Applying a **confidence threshold** to filter unreliable pseudo-labels.



## Installation

Clone the repository:

```bash
git clone https://github.com/your-repo/sim2real-mean-teacher.git
cd sim2real-mean-teacher
```

Install dependencies:

```bash
pip install -r requirements.txt
```



## Running the Code

Training can be started with:

```bash
python3 main.py --config config.yaml
```

Configuration options such as:

- dataset paths
- batch size
- learning rate
- other hyperparameters

can be modified in `config.yaml`.

## Notes 
- **Multi-GPU / distributed training is not implemented** in this repository.

## References

Tarvainen, A., & Valpola, H. (2017).  
**Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results.**  
Advances in Neural Information Processing Systems (NeurIPS).


## Acknowledgements

- Mean Teacher framework by **Tarvainen & Valpola**
- **RPTU Forest Dataset**

## Todos
- Dataset release
