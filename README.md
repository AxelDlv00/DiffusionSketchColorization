# Sketch Colorization Using Diffusion Models & Photo-Sketch Correspondence

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/AxelDlv00/DiffusionSketchColorization/tree/main)  [![Hugging Face Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface)](https://huggingface.co/datasets/ComputerVisionAnimeProject/AnimeFaceColorization/blob/main/README.md)

## Overview
This project explores **anime sketch colorization** using state-of-the-art **diffusion models** and **photo-sketch correspondence techniques**. Inspired by recent advancements in **AnimeDiffusion**, **MangaNinja**, and **photo-sketch correspondence models**, our method is a lighter model.

**Read the Full Paper:** [Sketch Colorization Using Diffusion Models](./sketch_colorization.pdf)

## Datasets Used
We created a **curated dataset** combining:
- **AnimeFace** [[Dataset](https://www.kaggle.com/datasets/thedevastator/anime-face-dataset-by-character-name)]
- **Danbooru Tagged Dataset** [[Dataset](https://www.kaggle.com/datasets/mylesoneill/tagged-anime-illustrations)]
- **[AnimeDiffusion Dataset](https://xq-meng.github.io/projects/AnimeDiffusion/)** augmented with **deformation flows**  

## Getting Started
### Installation
To set up the environment, install dependencies using:

```bash
git clone https://github.com/AxelDlv00/DiffusionSketchColorization.git
cd DiffusionSketchColorization
```

## Citing
If you use this model, please cite:
```
@misc{delavalkoita2025sketchcolorization,
  author       = {Axel Delaval and Adama Koïta},
  title        = {Sketch Colorization Using Diffusion Models and Photo-Sketch Correspondence},
  year         = {2025},
  url          = {https://github.com/AxelDlv00/DiffusionSketchColorization},
  note         = {Project exploring anime sketch colorization using diffusion models and deep learning.}
}
```
