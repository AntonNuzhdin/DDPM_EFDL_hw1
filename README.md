# Week 2 Home Assignment  

See the WandB logs here: [WandB Logs](https://wandb.ai/anthonnuzhdin/EFDL_DDPM_HW1)  

---

## 🐛 Bugs in the Original Code  

### **1. `diffusion.py`**  
These bugs were identified through test runs, code reviews, and comparisons with the original paper.  

- **`forward` method:**  
  - Move `timestep` to the correct device.  
  - Incorrect formula for `x_t` (should use `sqrt_one_minus_alpha_prod` in the original VP-SDE process).  
  - `eps` should be sampled from `N(0, I) → torch.randn_like()`.  

- **`sample` method:**  
  - Move `x_i` and `z` to the correct device.  

- **`get_schedules` method:**  
  - Added assertion: `0 < beta1 < beta2 < 1.0`.  
  - Ensure `betas` are positive numbers.  

---

### **2. `unet.py`**  
- **`forward` method:**  
  - `temb` should be expanded with two fictitious dimensions to enable broadcasting.  
  - This issue was found while running initial tests.  

---

### **3. `tests/`**  
- Added a fixed seed during test execution to ensure reproducibility.  
- This was necessary due to a flapping test and is a common best practice in all experiments.  

---

## ✅ Other Modifications  

### **🧪 Testing Coverage**  

```
coverage: platform darwin, python 3.12.4-final-0
```

| Name                      | Stmts | Miss | Cover |
|---------------------------|-------|------|-------|
| modeling/__init__.py      | 0     | 0    | 100%  |
| modeling/diffusion.py     | 34    | 0    | 100%  |
| modeling/training.py      | 30    | 0    | 100%  |
| modeling/unet.py          | 68    | 0    | 100%  |
| **TOTAL**                 | 132   | 0    | 100%  |

---

### **🔧 Additional Improvements**  

- **Added `test_training` in `tests/test_pipeline`** using `pytest`.  
  - This test covers the entire training process.  

- **Refactored `main.py`** to work seamlessly with **Hydra**.  
- **Added `logger/writer.py`**:  
  - Implemented a `wandb` class for logging experiments.  
  - Enabled logging of images and other useful information.  

- **Enhanced logging in `main.py`**:  
  - Logs all hyperparameters and metrics to `wandb`.  
  - Modified `train_epoch` and `generate_samples` to align with logging logic.  

- **Integrated DVC and configured it with Hydra**.  
- **Added a `pyproject.toml` file** with all dependencies.  
- **Included `uv.lock` for dependency management**.  
- **Set the random seed before training** to ensure experiment reproducibility.  

---

## ⚙️ Installation  

### **1️⃣ Install the `uv` Package Manager**  
```sh
pip install uv
```

### **2️⃣ Install Dependencies**  
```sh
uv pip install -r pyproject.toml
```

### **3️⃣ Configure DVC for Hydra Integration**  
```sh
dvc config hydra.enabled True
```

### **4️⃣ Manage Hyperparameters**  
- Hyperparameters for the model and training process are located in the **`configs/`** folder.  
- You can also view all parameters in **`params.yaml`**, which is auto-generated by DVC during each experiment.  

### **5️⃣ Run the Default Training Pipeline**  
```sh
uv run dvc exp run
```
---

🔥 Now you're ready to train and experiment! 🚀  
