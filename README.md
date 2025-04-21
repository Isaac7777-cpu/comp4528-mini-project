# comp4528-mini-project

This project is about single image generation. There is a recent improvement of the visual auto-regressive model which proposes the change in the perspective of viewing the problem by changing how the input token is constructed to the transformer model. This idea is particularly aligned with the original SinGAN, in which they have used Generative Adversarial model (GAN) to generate different scale of the image iteratively. However, this idea is soon improved by employing using the diffusion model instead. Therefore, I am trying to use a decoder in the transformer to achieve a similar result.

## Other Artifacts

During my exploration of the idea, I have also implemented multiple other ideas as well, including vae and vqvae which are used extensively for VAR as well.
