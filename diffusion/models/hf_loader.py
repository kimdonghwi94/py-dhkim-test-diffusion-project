from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from typing import List, Union
from enum import Enum

import torch
import numpy as np
import PIL

#TODO 삭제예정

PipelineImageInput = Union[
    PIL.Image.Image,
    np.ndarray,
    torch.FloatTensor,
    List[PIL.Image.Image],
    List[np.ndarray],
    List[torch.FloatTensor],
]


class ModelType(str, Enum):
    Img2Img = 'txt2img'
    Txt2Img = 'img2img'
    Inpaint = 'inpaint'


class DiffusionModel:
    def __init__(self, path, device = 'cuda'):
        self.path = path
        self.device = device if torch.cuda.is_available() else 'cpu'

    def load_txt2img(self):
        """
        Users/donghwi/PycharmProjects/py-did-project-diffusion/venv/Lib/site-packages/diffusers/pipelines/stable_diffusion_xl/__init__

        """
        path = self.path
        device = self.device
        stable_diffusion_txt2img = DiffusionPipeline.from_pretrained(
            path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        stable_diffusion_txt2img.to(device)

        self.stable_diffusion_txt2img = stable_diffusion_txt2img

    def load_refiner(self):
        path = self.path
        device = self.device
        stable_diffusion_txt2img = self.stable_diffusion_txt2img
        refiner = DiffusionPipeline.from_pretrained(
            path,
            text_encoder_2=stable_diffusion_txt2img.text_encoder_2,
            vae=stable_diffusion_txt2img.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        refiner.to(device)

        self.refiner = refiner

    def load_img2img(self):
        """
        https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img
        """
        stable_diffusion_txt2img = self.stable_diffusion_txt2img
        device = self.device
        self.stable_diffusion_img2img = StableDiffusionImg2ImgPipeline(
            vae=stable_diffusion_txt2img.vae,
            text_encoder=stable_diffusion_txt2img.text_encoder,
            tokenizer=stable_diffusion_txt2img.tokenizer,
            unet=stable_diffusion_txt2img.unet,
            scheduler=stable_diffusion_txt2img.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        self.stable_diffusion_img2img.to(device=device)

    def load_inpaint(self):
        """
        https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint
        """
        stable_diffusion_txt2img = self.stable_diffusion_txt2img
        device = self.device
        self.stable_diffusion_inpaint = StableDiffusionInpaintPipeline(
            vae=stable_diffusion_txt2img.vae,
            text_encoder=stable_diffusion_txt2img.text_encoder,
            tokenizer=stable_diffusion_txt2img.tokenizer,
            unet=stable_diffusion_txt2img.unet,
            scheduler=stable_diffusion_txt2img.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,)
        self.stable_diffusion_inpaint.to(device)

    def load_model(self, type: ModelType = ModelType.Txt2Img, refiner: bool = False):
        self.load_txt2img()
        self.load_txt2img()
        self.load_img2img()
        self.load_inpaint()
        if type == ModelType.Txt2Img and refiner:
            self.load_refiner()

    def __call__(self, **kwargs):
        return self.__getitem__(**kwargs)

    def __getitem__(self, model_type: ModelType, seed, prompt, **kwargs):
        torch.manual_seed(seed)
        if self.device =='cuda':
            torch.cuda.manual_seed_all(seed)

        if model_type == ModelType.Txt2Img:
            stable_diffusion_output = self._txt2img_call(prompt, **kwargs)
            if hasattr(self, 'refiner'):
                stable_diffusion_output = self._refiner_call(prompt, stable_diffusion_output)
        elif model_type == ModelType.Img2Img:
            stable_diffusion_output = self._img2img_call(prompt, **kwargs)
        elif model_type == ModelType.Inpaint:
            stable_diffusion_output = self._inpaint_call(prompt, **kwargs)
        else:
            return ''
        return stable_diffusion_output.images[0]

    def _refiner_call(self,
                      prompt: str = '',
                      stable_diffusion_output =None,
                      n_steps: int = 20,
                      high_noise_frac: float = 0.8,
                      ):
        image = stable_diffusion_output.images[0]

        refiner = self.refiner
        stable_diffusion_output = refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        )
        return stable_diffusion_output

    def _inpaint_call(self,
                      prompt: str='',
                      image: PipelineImageInput=None,
                      mask_image: PipelineImageInput=None,
                      height: int=None,
                      width: int=None,
                      negative_prompt: str = '',
                      n_steps: int = 20,
                      high_noise_frac: float = 0.8,
                      **kwargs
                      ):
        """

        Parameters
        ----------
        prompt
        image
        mask_image
        height
        width
        negative_prompt
        n_steps
        high_noise_frac
        kwargs

        Returns
        -------

        """
        stable_diffusion_inpaint = self.stable_diffusion_inpaint
        stable_diffusion_output = stable_diffusion_inpaint(prompt=prompt,
                                                           image=image,
                                                           mask_image=mask_image,
                                                           height=height,
                                                           width=width,
                                                           negative_prompt=negative_prompt,
                                                           num_inference_steps=n_steps,
                                                           denoising_end=high_noise_frac,
                                                           **kwargs)
        return stable_diffusion_output

    def _img2img_call(self,
                      prompt: str = '',
                      image: PipelineImageInput = None,
                      negative_prompt: str = '',
                      n_steps: int = 20,
                      high_noise_frac: float = 0.8,
                      **kwargs
                      ):
        """

        Parameters
        ----------
        prompt
        image
        negative_prompt
        n_steps
        high_noise_frac
        kwargs

        Returns
        -------

        """
        stable_diffusion_img2img = self.stable_diffusion_img2img
        stable_diffusion_output = stable_diffusion_img2img(prompt=prompt,
                                                           image=image,
                                                           negative_prompt=negative_prompt,
                                                           num_inference_steps=n_steps,
                                                           denoising_end=high_noise_frac,
                                                           **kwargs)
        return stable_diffusion_output


    def _txt2img_call(self,
                      prompt: str='',
                      negative_prompt: str='',
                      n_steps: int = 20,
                      high_noise_frac: float = 0.8,
                      height: int = 512,
                      width: int = 512,
                      **kwargs):
        """

        Parameters
        ----------
        prompt
        negative_prompt
        n_steps
        high_noise_frac
        kwargs

        Returns
        -------

        """
        stable_diffusion_txt2img = self.stable_diffusion_txt2img
        stable_diffusion_output = stable_diffusion_txt2img(prompt=prompt,
                                                           negative_prompt=negative_prompt,
                                                           num_inference_steps=n_steps,
                                                           denoising_end=high_noise_frac,
                                                           height=height,
                                                           width=width,
                                                           **kwargs)
        return stable_diffusion_output
