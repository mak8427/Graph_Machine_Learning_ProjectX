# Objectives

### Redefine the build_spatio_temporal_edge function

### Plot correctly the graphs

### Documentation part
# Questions

### Is the position of the nodes normalised?

### Are there new Papers 
- Back to MLP: A Simple Baseline for Human Motion Prediction: https://arxiv.org/pdf/2207.01567 

---
### What losses should we use and how to evaluate the model (Davide)

The SIMLPE model uses two types of loss functions for training human motion prediction:

#### 1. **Reconstruction Loss (\(L_{\text{re}}\))**
- Measures the **L2-norm** (Euclidean distance) between the predicted motion sequence (\(x'_{T+1:T+N}\)) and the ground truth motion sequence (\(x_{T+1:T+N}\)).
- Ensures that the predicted future poses closely match the ground truth poses.
- **Formula**:
  \[
  L_{\text{re}} = \|x'_{T+1:T+N} - x_{T+1:T+N}\|_2
  \]

#### 2. **Velocity Loss (\(L_v\))**
- Optimizes the velocity of the predicted motion sequence to match the velocity of the ground truth motion sequence.
- The velocity is defined as the difference between consecutive frames: \(v_t = x_{t+1} - x_t\).
- Helps maintain smooth transitions and captures dynamic patterns of motion.

**Formula**:
  \[
  L_v = \|v'_{T+1:T+N} - v_{T+1:T+N}\|_2
  \]

  - \(v'_{T+1:T+N}\): Velocity of the predicted motion sequence.
  - \(v_{T+1:T+N}\): Velocity of the ground truth sequence.

#### Combined Loss Function
The overall loss used during training combines both reconstruction and velocity losses:
\[
L = L_{\text{re}} + L_v
\]

- The **reconstruction loss** ensures accurate positional predictions for the joints.
- The **velocity loss** ensures smooth and realistic motion dynamics over time.
- Including the velocity loss improves the model's performance, especially for **long-term predictions**, as validated in the paper's ablation studies.



### How do we differentiate between temporal and spacial features (Pablo)
- Positional embedding
- 