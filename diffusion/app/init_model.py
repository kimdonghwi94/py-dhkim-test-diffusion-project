from diffusion.models import hf_loader


DiffusionModel = hf_loader.DiffusionModel(r'stabilityai/stable-diffusion-xl-base-1.0')
DiffusionModel.load_model()