import os

from diffusion.web.domain.response import CacheResponse,CacheData,CacheResult,CacheInfo
from diffusion.models.hf_loader import PipelineImageInput

from pathlib import Path
from typing import Union

import PIL
import hashlib
import time
import json


def save_file(tree, param, image: PipelineImageInput):
    file_date = time.strftime('%Y%m%d%H')
    cache_date = time.strftime('%Y-%m-%d-%H-%M-%S')
    byte_hash = hash(image)

    image_save_path = save_image(tree, param, file_date, byte_hash, image)
    save_cache(tree, param, file_date, cache_date, byte_hash, image_save_path)

    return relpath(image_save_path, cwd=tree.path(tag='data'))


def hash(img):
    # img = img.getdata()
    return hashlib.sha256(img.tobytes()).hexdigest()


def save_image(tree, param, file_date, hash, image):
    image_path = tree.vpath(tag='diffusion_image')
    image_save = image_path.path({'type': param.model_type.value,
                                  'date': file_date,
                                  'hash': hash})
    image_save = check_file(image_save.parent, image_save.name, True)
    if type(image) == PIL.Image.Image:
        image.save(str(image_save), 'JPEG')

    return image_save


def save_cache(tree, param, file_date, cache_date, hash, image_save_path):
    cache_path = tree.vpath(tag='diffusion_cache')

    cache_file_path = cache_path.path({'type': param.model_type.value,
                                       'date': file_date,
                                       'hash': hash})

    info = CacheInfo(task_name=hash,
                     times=cache_date,
                     version='0.1.0',
                     model_name=param.model_type,
                     inputs=param.dict(),
                     seed=param.seed)
    data = CacheData(res=image_save_path)
    result = CacheResult(info = info, data=data)
    response = CacheResponse(result = result)
    cache_file_path = check_file(cache_file_path.parent, cache_file_path.name, True)
    _write(cache_file_path, response.dict())


def _write(path: str, data):
    """
    json 파일을 생성한다.

    Parameters
    ----------
    path: str
        파일 경로
    data: dict
        저장할 데이터

    Returns
    -------
    bool
        파일 생성 여부
    """
    # 파일 열기
    with open(path, 'w', encoding='utf8') as f:
        # pickle 파일 저장
        json.dump(data, f, indent=4)


def is_file(in_path: str) -> bool:
    """
    데이터 유무 확인

    Parameters
    ----------
    in_path: str
        파일 경로

    Returns
    -------
    bool
        경로 존재 유무 반환

    Examples
    --------
    >>> is_file('test/path/test.jpg')
    False
    """
    return os.path.isfile(in_path)


def is_dir(in_path: str, exist: bool = False) -> bool:
    """
    폴더 유무 확인
    exist 폴더 존재 하지 않으면 생성

    Parameters
    ----------
    in_path: str
            입력 데이터 경로
    exist: bool
         폴더 생성. Default False

    Returns
    -------
    bool
        폴더 존재 유무 반환

    Examples
    --------
    >>> is_dir('test/path/test.jpg')
    False
    >>> is_dir('test/path/test.jpg', exist=True)
    True
    """
    if os.path.isdir(in_path):
        return True

    if not exist:
        return False

    os.makedirs(in_path, exist_ok=True)
    return True


def split_path_name(in_path: Union[str, Path]) -> tuple:
    """
    경로, 이름(포맷) 반환

    Parameters
    ----------
    in_path: str

    Examples
    --------
    >>> split_path_name('test/path/test.jpg')
    ('tset/path', 'test.jpg')

    """
    path, name = os.path.split(in_path)
    return path, name


def split_name_ext(in_path: Union[Path, str]) -> tuple:
    """
    이름, 포맷 반환

    Parameters
    ----------
    in_path: str

    Examples
    --------
    >>> split_name_ext('test/path/test.jpg')
    ('test', '.jpg')

    """
    _, name = split_path_name(in_path)
    name, ext = os.path.splitext(name)
    return name, ext.lower()


def check_file(path: str, filename: str, check_extist: bool = False):
    is_dir(path, exist=True)
    if check_extist:
        filepath = os.path.join(path, filename)
        if is_file(filepath):
            number = 0
            name, ext = split_name_ext(filename)
            while is_file(filepath):
                number += 1
                filename = f'{name}({number}){ext}'
                filepath = os.path.join(path, filename)
            # filename = name + f"{len(os.listdir(path)) + 1}" + ext
    return os.path.join(path, filename)