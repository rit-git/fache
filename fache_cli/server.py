import hydra
import uvicorn
from omegaconf import DictConfig

from fache.api import FacheAPI

@hydra.main(config_path="../conf", config_name="inference_config", version_base=None)
def main(cfg: DictConfig):
    app = FacheAPI(cfg)
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=cfg.server.port if 'server' in cfg else 12345,
        root_path='/app/fache'
    )

if __name__ == '__main__':
    main()
