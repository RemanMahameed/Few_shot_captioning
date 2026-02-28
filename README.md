# Pokémon Image Captioning: Parameter-Efficient Fine-Tuning with LoRA

## Overview
This project explores parameter-efficient fine-tuning (PEFT) of the **Microsoft GIT** (Generative Image2Text) vision-language model for the task of image captioning. Using a subset of Pokémon images, the model is fine-tuned to generate descriptive captions based on visual inputs while maintaining a minimal memory footprint.

## Dataset
* **Source:** `reach-vb/pokemon-blip-captions`
* **Subset:** 700 images (Split: 80% Training, 10% Validation, 10% Testing)
* **Processing:** Images and text are processed using `AutoProcessor`, with labels explicitly masked for padding tokens (`-100`) to ensure accurate loss calculation without penalizing padding.

## Model & Architecture
* **Base Model:** `microsoft/git-base`
* **Fine-Tuning Strategy:** Low-Rank Adaptation (LoRA)
* **Architecture Choice:** A Text-Only LoRA configuration was used to maximize parameter efficiency and maintain stability. The applied LoRA targets specific attention and dense modules: `query`, `key`, `value`, and `dense`. 

### LoRA Configuration
We carefully tuned the LoRA capacity to balance expressiveness and generalization on our small dataset:
* **Rank (r = 16) & Alpha (α = 32):** We selected r=16 and α=32 as our optimal configuration. Because the LoRA weight update (ΔW) is scaled by α/r, decreasing the rank from 32 to 16 while holding α constant effectively increases the scaling factor from 1 to 2. This yields stronger updates per learned direction while keeping the trainable parameter count low, providing an excellent bias-variance trade-off for the low-data regime.
* **Dropout:** LoRA dropout is enabled to regularize the adapter pathway and reduce the risk of overfitting.

## Training Hyperparameters
The model was trained using the Hugging Face `Trainer` API with the following specific hyperparameter choices:
* **Epochs (15):** Ensures sufficient exposure to the small dataset. Validation loss decreases rapidly early on and improves gradually thereafter. Best-checkpoint selection mitigates the risk of overfitting in the later epochs.
* **Learning Rate (5e-5):** A moderately higher learning rate proved beneficial for adapting the frozen backbone from limited data, producing more coherent captions than smaller rates.
* **Batch Size & Gradient Accumulation:** To manage transformer memory constraints, we used a per-device batch size of 2 combined with 4 gradient accumulation steps, yielding a stable **Effective Batch Size of 8**.
* **Warmup Ratio (0.05):** The learning rate is linearly increased over the first 5% of training steps to prevent early instability.
* **Weight Decay (0.0):** Because LoRA already restricts capacity, strong weight regularization was omitted to allow the adapter to fully capture dataset-specific patterns.
* **Evaluation Strategy:** Checkpoints were evaluated and saved every epoch (`eval_strategy="epoch"`), utilizing `load_best_model_at_end=True` to automatically select the checkpoint with the lowest validation loss.

## Evaluation & Metrics
The model was evaluated using two primary metrics:
**Cross-Entropy Loss:** Tracked across both training and validation sets to monitor convergence and stability.

### Current Limitations
Due to the aggressively constrained dataset size (700 images), the model successfully learns domain-specific vocabulary (e.g., "drawing", "bird", "cartoon") but currently lacks the extensive structural exposure needed for complex grammar. This occasionally results in scrambled syntax during generation for longer text sequences.

## How to Use (Google Colab)
The easiest way to test or train this model is directly through Google Colab. 

1. **Open the Notebook:** Click the "Open in Colab" badge at the top of this README, or open `few_shot_captioning_testing_LoRA_T.ipynb` directly in Colab.
2. **Enable GPU:** Go to `Runtime` > `Change runtime type` and ensure a hardware accelerator like a **T4 GPU** is selected.
3. **Execute:** Go to `Runtime` > `Run all`. 
   * *Note:* The first cell will automatically handle installing all necessary dependencies (`evaluate`, `jiwer`, `peft`, `datasets`, `transformers`).
4. **View Inference:** Scroll to the bottom of the notebook to see the plotted evaluation metrics and the model generating hyperparameter-tuned captions for the validation images.
