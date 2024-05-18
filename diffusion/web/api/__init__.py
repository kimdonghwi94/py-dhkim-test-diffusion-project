from diffusion.modules.sd_models import model_path
from diffusion.modules.modelloader import load_models
from diffusion.web.domain.request import Text2ImageParam, InpaintParam, Image2ImageParam
from diffusion.web.domain.response import StableDiffusionResponse
from diffusion.app.init_model import DiffusionModel
from diffusion.resources import path as paths
from diffusion.app.util import save_file

from diffusion import web

from fastapi import APIRouter, Depends

import torch

seed = -1

root_path = rootpath(__file__, cwd=web)
router = APIRouter(prefix=root_path, tags=[root_path])

# 환경 설정 파일
env = EnvConfLoader(paths.abspath('path.env'))
# 디렉토리
tree = VariablePathTree(tags=env.conf)


@router.get('/models/list')
def list_models():
    return load_models(model_path=model_path)


@router.get('/models/')
def is_load():
    pass


@router.post('/models/txt2img', response_model=StableDiffusionResponse)
def generate_txt2img(param: Text2ImageParam = Depends(Text2ImageParam)):
    torch.manual_seed(param.seed)
    if param.refiner:
        DiffusionModel.load_refiner()
    image = DiffusionModel(**param.dict())
    res = save_file(tree, param, image)
    return StableDiffusionResponse(res=res)


@router.post('/models/img2img', response_model=StableDiffusionResponse)
def generate_img2img(param: Image2ImageParam = Depends(Image2ImageParam)):
    torch.manual_seed(param.seed)
    image = DiffusionModel(**param.dict())
    res = save_file(tree, param, image)
    return StableDiffusionResponse(res=res)


@router.post('/models/inpaint', response_model=StableDiffusionResponse)
def generate_inpaint(param: InpaintParam = Depends(InpaintParam)):
    torch.manual_seed(param.seed)
    image = DiffusionModel(**param.dict())
    res = save_file(tree, param, image)
    return StableDiffusionResponse(res=res)
