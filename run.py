import sys

sys.path.append("cog-flux")

import torch
from predict import DevPredictor, SchnellPredictor

torch.cuda.empty_cache()


predictor = SchnellPredictor()

predictor.setup()
output = predictor.predict(
    prompt="A beautiful image of a cat",
    aspect_ratio="1:1",
    # image=None,
    num_outputs=1,
    seed=None,
    output_format="jpg",
    num_inference_steps=4,
    output_quality=80,
    disable_safety_checker=True,
    go_fast=True,
    megapixels="1",
)

print(output)
