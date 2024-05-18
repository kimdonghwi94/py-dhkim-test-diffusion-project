from pydantic import BaseModel, Field

class StableDiffusionResponse(BaseModel):
    res: str = Field(description='', default='')

    def __str__(self):
        return type(self).__name__


class CacheInfo(BaseModel):
    task_name: str = Field(default='')
    times: str = Field(default='')
    version: str = Field(default='')
    model_name: str = Field(default='')
    seed: int = Field(-1)
    inputs: dict = Field({})


class CacheData(BaseModel):
    res: str = Field(default={})


class CacheResult(BaseModel):
    info: CacheInfo = Field(default=CacheInfo())
    data : CacheData = Field(default=CacheData())


class CacheResponse(BaseModel):
    result: CacheResult = Field(default=CacheResult())