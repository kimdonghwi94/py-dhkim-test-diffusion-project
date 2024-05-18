from qmc_api.service import add_health, add_endpoints, ModFastAPI
from fastapi import FastAPI
from diffusion.web.api import router
import diffusion

app: FastAPI = ModFastAPI(diffusion)

# /web/v1
app.include_router(router)

add_health(app)
add_endpoints(app)


if __name__ == '__main__':
    import os
    import sys
    import uvicorn

    os.chdir(diffusion.path.parent)
    if '--port' in sys.argv:
        __i__ = sys.argv.index('--port') + 1
        port = int(sys.argv[__i__])
    else:
        port = 8003

    uvicorn.run(app, port=port, log_config=diffusion.path.abspath('resources', 'log.ini'))
