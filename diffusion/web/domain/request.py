from pydantic import BaseModel, Field, validator
from typing import Union

from diffusion.models.hf_loader import ModelType
from diffusion.app.init_path import validator_di


import os


class DiffusionParam(BaseModel):
    model_type: ModelType = Field(default=ModelType.Txt2Img, choices=[ModelType.Img2Img, ModelType.Img2Img, ModelType.Inpaint])
    prompt: str = Field(default='')
    negative_prompt: str = Field(default='')
    seed: int = Field(default=-1)

    @validator('prompt')
    def validate_prompt(cls, txt: str):
        return txt

    @validator('negative_prompt')
    def validate_negative_prompt(cls, txt: str):
        return txt

    def __str__(self):
        return type(self).__name__


class Text2ImageParam(DiffusionParam):
    n_steps: int = Field(description='',default=20)
    high_noise_frac: float = Field(description='',default=0.8)
    height: int = Field(default=512)
    width:int = Field(default=512)
    refiner: bool = Field(default=False)


class Image2ImageParam(DiffusionParam):
    image: str = Field(default='')
    n_steps: int = Field(description='', default=20)
    high_noise_frac: float = Field(description='', default=0.8)

    @validator('image')
    def validate_image(cls, path: str):
        data_path = validator_di.tree.path(tag='data')
        path = os.path.join(data_path, path)

        if not os.path.exists(path):
            raise validator_di.err_mgr.err()
        path.replace('\\','/')
        return path


class InpaintParam(DiffusionParam):
    image: str = Field(default='')
    mask_image: str = Field(default='')
    height: int = Field(default=512)
    width: int = Field(default=512)
    n_steps: int = Field(description='', default=20)
    high_noise_frac: float = Field(description='', default=0.8)

    @validator('image')
    def validate_image(cls, path: str):
        data_path = validator_di.tree.path(tag='data')
        path = os.path.join(data_path, path)

        if not os.path.exists(path):
            raise validator_di.err_mgr.err()
        path.replace('\\', '/')
        return path

    @validator('mask_image')
    def validate_mask_image(cls, path: str):
        data_path = validator_di.tree.path(tag='data')
        path = os.path.join(data_path, path)

        if not os.path.exists(path):
            path = ''
        return path

