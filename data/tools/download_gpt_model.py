from tools.log import log
from data.tools.gpt_download import download_and_load_gpt2
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(project_root, "output", "gpt2")
os.makedirs(output_dir, exist_ok=True)

settings, params = download_and_load_gpt2(
    model_size="124M", models_dir=output_dir
)
log.info("Settings: %s", settings)
log.info("Parameter dictionary keys: %s", list(params.keys()))

log.info(params["wte"])
log.info("Token embedding weight tensor dimensions: %s", params["wte"].shape)

