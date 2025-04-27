# 🌙 CVAE-Guided Color Restoration for Low-Light Image Enhancement

Welcome to our project on **low-light image enhancement (LLIE)** with **Conditional Variational Autoencoder (CVAE)** guidance!
We focus on improving color restoration in extremely dark regions where traditional enhancement models fail due to the absence of reliable lighting cues.

---


# Dataset

Unzip and put dataset [LOL-V2](https://drive.google.com/drive/folders/1zoBcUq5o0VmuXHd-9w1wgTby4183Yd-x?usp=drive_link) and [LOL-V1](https://drive.google.com/file/d/157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB/view) under the LLIE_Project folder if you want to try out the model.

# Training 

run the following code to train the model, the configuration file is in `./Options`. 

```python
python main.py --model_option ./Options/RetinexFormer_custom.yml --train_save_dir CUSTOM_SAVE_DIR
```

# Pretrained Result

There are already four model being trained and stored in `./train_results`
- `./train_results/reinexformer_baseline` & `./train_results/reinexVAE` are trained on multiple different dataset.
- `./train_results/reinexformer_baseline_onlyLOLV2` & `./train_results/reinexVAE_onlyLOLV2` are trained only on LOLV2 Normal dataset.

---

# 📝 References

Zhou, N., Li, C., Huang, Y., Guo, C., Feng, W., Loy, C. C., & Gu, J. (2023). RetinexFormer: One-stage Retinex-based Transformer for Low-light Image Enhancement. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

Guo, C., Li, C., Guo, J., Loy, C. C., Hou, J., Kwong, S., & Cong, R. (2020). Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 1780–1789.
