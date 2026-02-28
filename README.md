# Pokémon Image Captioning: Parameter-Efficient Fine-Tuning with LoRA

## Overview
Parameter-efficient fine-tuning (PEFT) of the **Microsoft GIT** vision-language model for image captioning. This project uses a Text-Only LoRA approach on a small Pokémon dataset to generate captions while keeping compute requirements minimal.

## Dataset
* **Source:** `reach-vb/pokemon-blip-captions`
* **Subset:** 700 images (80% Train / 10% Val / 10% Test)
* **Processing:** Handled via `AutoProcessor`. Padding tokens are masked with `-100` to ensure accurate loss calculation.

## Architecture & LoRA Configuration
* **Base Model:** `microsoft/git-base`
* **Strategy:** Text-Only LoRA (Targeting `query`, `key`, `value`, `dense`).
* **Why Text-Only?** A dual-encoder setup showed signs of overfitting. Text-only achieved nearly identical validation loss (1.704) with half the trainable parameters.
* **LoRA Parameters:** `r=16`, `α=32`, with dropout enabled. 
  * *Note:* Decreasing the rank to 16 while holding α at 32 increases the scaling factor (α/r) from 1 to 2. This delivers stronger updates per learned direction while using fewer parameters, providing an ideal bias-variance trade-off for a small dataset.

## Training Hyperparameters
* **Epochs:** 15 (with `load_best_model_at_end=True` to select the checkpoint with the lowest validation loss)
* **Learning Rate:** 5e-5 (Provides better adaptation from limited data than smaller rates)
* **Effective Batch Size:** 8 (Per-device batch size of 2 × 4 gradient accumulation steps)
* **Warmup Ratio:** 0.05
* **Weight Decay:** 0.0 (LoRA naturally restricts capacity; strong regularization was unnecessary)

## Evaluation 
**Cross-Entropy Loss:** Evaluated every epoch to monitor convergence. Best-checkpoint selection relies purely on minimizing validation loss, avoiding the last epoch if it begins to overfit.

**Current Limitations:** Given the tiny dataset (700 images), the model quickly learns domain vocabulary ("bird", "drawing", "cartoon") but lacks the structural exposure for complex grammar, occasionally leading to scrambled syntax on longer captions.

## How to Use (Google Colab)
1. **Open the Notebook:** Click the "Open in Colab" badge above.
2. **Enable GPU:** Go to `Runtime` > `Change runtime type` and select a **T4 GPU**.
3. **Execute:** Click `Runtime` > `Run all`. The first cell installs all dependencies (`peft`, `datasets`, `transformers`, etc.).
4. **View Inference:** Scroll to the bottom to see training/validation loss plots and generated captions for validation images.
