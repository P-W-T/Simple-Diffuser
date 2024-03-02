# Simple Diffusion
## Introduction
This repository is made to make diffusion accesible. The implementation of diffusion presented here is supposed to be simple, clear and lightweight:
- Simple: The concepts are build up slowly (more complex principles are added on top of a solid base) and complications are kept to a minimum.
- Clear: The code is well annotated and is all in one file. This is a complete, functional diffusion model in 100s of lines not 1000s.
- Lightweight: This code does not require expensive hardware. I did this on a single secondhand RTX3090 with room to spare, by changing the batch size, this can be trained on cheaper GPUs.

## Dataset
The current code uses Fashion MNIST (https://github.com/zalandoresearch/fashion-mnist) to keep it easy and fast. In the future I would like to test other (more challenging) datasets.

## Files
### Diffusion evaluation
- Create Classifier(FM).ipynb: Trains a classifier that is used to calculate the Inception score and the Fréchet inception distance (FID) score
- diffusion_score.py: The code needed for evaluating the diffusion model. This file contains code to train and evaluate a classifier and code to use the classifier to calculate the inception score. Code is also foreseen to calculate the Fréchet inception distance score

## Diffusion
- simple_diffuser.py: The base diffuser, designed based on the code of https://www.youtube.com/watch?v=TBCRlnwJtZU, with a lot more annotation
- simple_diffuser_improved.py: The base diffuser from simple_diffuser.py but with improved flexibility built in into the UNet allowing the reuse of the code for many applications
- simple_diffuser_improved_cosine.py: The base improved diffuser but with added noise schedulers such as cosine scheduling (which performs better with smaller images)
- simple_diffuser_improved_cosine_CFG.py: The base improved diffuser with additional classifier free guidance (CFG) allowing conditional image generation
- simple_diffuser_improved_cosine_CFG_EMA.py: The base improved diffuser with additional exponential moving average (EMA) training allowing a smoother training
- Simple Diffuser base(FM).ipynb: Jupyter notebook to train diffusers from this repository
- Simple Diffuser base(FM) - conditional.ipynb: Jupyter notebook to train diffusers from this repository with conditional image generation

## Reference
The code is inspired by the excellent videos by Outlier (https://www.youtube.com/@outliier), and his explanation of diffusion models is excellent. Next to that, the code and improvements are based on some landmark papers that can be found in the references.
- Tutorial explanation: https://www.youtube.com/watch?v=HoKDTa5jHvg&t=1594s
- Coding tutorial: https://www.youtube.com/watch?v=TBCRlnwJtZU
- Annotated diffuser: https://huggingface.co/blog/annotated-diffusion
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. Advances in neural information processing systems, 33, 6840-6851.
- Nichol, A. Q., & Dhariwal, P. (2021, July). Improved denoising diffusion probabilistic models. In International Conference on Machine Learning (pp. 8162-8171). PMLR.
- Ho, J., & Salimans, T. (2022). Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598.- 

