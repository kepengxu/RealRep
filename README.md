# RealRep: Generalized SDR-to-HDR Conversion via Attribute-Disentangled Representation Learning

**Li Xu¹, Siqi Wang¹, Kepeng Xu¹✉️, Lin Zhang, Gang He¹, Weiran Wang¹, Yu-Wing Tai²**

¹Xidian University, ²Dartmouth College

> **Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2026 (Oral Presentation)**

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/pdf/2505.07322v3) [![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://kepengxu.github.io/projects/realrep/realrep.html)

---

## 📢 News
* **[2025/12]** We are prepared the code and pretrained models. The Code have been released! 🚀
* **[2025/12]** This paper has been accepted by **AAAI 2026** as an **Oral Presentation**! 🎉

## 🏠 Abstract
High-Dynamic-Range Wide-Color-Gamut (HDR-WCG) technology is becoming increasingly widespread, driving a growing need for converting Standard Dynamic Range (SDR) content to HDR. Existing methods primarily rely on fixed tone mapping operators, which struggle to handle the diverse appearances and degradations commonly present in real-world SDR content.

To address this limitation, we propose a generalized SDR-to-HDR framework that enhances robustness by learning attribute-disentangled representations. Central to our approach is **RealRep**, which explicitly disentangles luminance and chrominance components to capture intrinsic content variations across different SDR distributions. Furthermore, we design a Luma-/Chroma-aware negative exemplar generation strategy that constructs degradation-sensitive contrastive pairs. Building on these attribute-level priors, we introduce the **Degradation-Domain Aware Controlled Mapping Network (DDACMNet)**, a lightweight, two-stage framework that performs adaptive hierarchical mapping guided by a control-aware normalization mechanism.

## 🚀 Methodology
* **Figure 1:** Comparison between previous entangled frameworks and our attribute-disentangled method.
* **Figure 2:** Overview of the DDACMNet architecture. It consists of multi-view encoders, a fusion module, and a controlled mapping network.

## 📊 Results

### Quantitative Comparison
Our method achieves state-of-the-art performance on the HDRTV4K dataset benchmarks.

| Method | Average PSNR | Average SSIM |
| :--- | :---: | :---: |
| HDRTVNet | 25.77 | 0.8716 |
| LSNet | 28.46 | 0.8979 |
| PromptIR | 28.34 | 0.8940 |
| **RealRep (Ours)** | **31.05** | **0.9219** |

### Qualitative Comparison
RealRep demonstrates superior stability under unknown degradations, eliminating artifacts and accurately recovering highlight details and vivid colors.

## 🔧 Usage
*(Code coming soon...)*

## 📝 Citation
If you find this project useful for your research, please consider citing:

```bibtex
@inproceedings{xu2026realrep,
  title={RealRep: Generalized SDR-to-HDR Conversion via Attribute-Disentangled Representation Learning},
  author={Xu, Li and Wang, Siqi and Xu, Kepeng and Zhang, Ling and He, Gang and Wang, Weiran and Tai, Yu-Wing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```
## 📧 Contact
If you have any questions, please contact Kepeng Xu at kepengxu11@gmail.com.
