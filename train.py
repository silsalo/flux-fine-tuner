import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import OrderedDict
from zipfile import ZipFile, is_zipfile

import yaml
from dotenv import load_dotenv

sys.path.append("ai-toolkit")

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from jobs import BaseJob
from toolkit.config import get_config

load_dotenv()

JOB_NAME = "flux-lora-train"
WEIGHTS_PATH = Path("./FLUX.1-dev")
INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("output")
JOB_DIR = OUTPUT_DIR / JOB_NAME


class CustomSDTrainer(SDTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_samples = set()
        self.wandb = None

    def hook_train_loop(self, batch):
        loss_dict = super().hook_train_loop(batch)
        return loss_dict

    def sample(self, step=None, is_first=False):
        super().sample(step=step, is_first=is_first)
        pass

    def post_save_hook(self, save_path: Path):
        super().post_save_hook(save_path)
        pass


class CustomJob(BaseJob):
    def __init__(self, config: OrderedDict, wandb_client: None):
        super().__init__(config)
        self.device = self.get_conf("device", "cpu")
        self.process_dict = {"custom_sd_trainer": CustomSDTrainer}
        self.load_processes(self.process_dict)
        for process in self.process:
            process.wandb = wandb_client

    def run(self):
        super().run()
        # Keeping this for backwards compatibility
        print(
            f"Running  {len(self.process)} process{'' if len(self.process) == 1 else 'es'}"
        )
        for process in self.process:
            process.run()


def clean_up():
    # TODO: logout_wandb()

    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


def train(
    input_images: Path,  # "A zip file containing the images that will be used for training. We recommend a minimum of 10 images. If you include captions, include them as one .txt file per image, e.g. my-photo.jpg should have a caption file named my-photo.txt. If you don't include captions, you can use autocaptioning (enabled by default).",
    trigger_word: str,  # "The trigger word refers to the object, style or concept you are training on. Pick a string that isn’t a real word, like TOK or something related to what’s being trained, like CYBRPNK. The trigger word you specify here will be associated with all images during training. Then when you use your LoRA, you can include the trigger word in prompts to help activate the LoRA.",
    steps: int = 1000,  # "Number of training steps. Recommended range 500-4000",
    learning_rate: float = 4e-4,  # Learning rate, if you’re new to training you probably don’t need to change this
    batch_size: int = 1,  # Batch size, you can leave this as 1
    resolution: str = "512,768,1024",  # Image resolutions for training
    lora_rank: int = 16,  # Higher ranks take longer to train but can capture more complex features. Caption quality is more important for higher ranks.
    caption_dropout_rate: float = 0.05,  # Advanced setting. Determines how often a caption is ignored. 0.05 means for 5% of all steps an image will be used without its caption. 0 means always use captions, while 1 means never use them. Dropping captions helps capture more details of an image, and can prevent over-fitting words with specific image elements. Try higher values when training a style.
    optimizer: str = "adamw8bit",  # Optimizer to use for training. Supports: prodigy, adam8bit, adamw8bit, lion8bit, adam, adamw, lion, adagrad, adafactor.
    cache_latents_to_disk: bool = False,  # Use this if you have lots of input images and you hit out of memory errors
    layers_to_optimize_regex: str = None,  # Regular expression to match specific layers to optimize. Optimizing fewer layers results in shorter training times, but can also result in a weaker LoRA. For example, To target layers 7, 12, 16, 20 which seems to create good likeness with faster training (as discovered by lux in the Ostris discord, inspired by The Last Ben), use `transformer.single_transformer_blocks.(7|12|16|20).proj_out`.",
    wandb_api_key: str = None,  # Weights and Biases API key, if you'd like to log training progress to W&B.
    wandb_project: str = JOB_NAME,  # Weights and Biases project name. Only applicable if wandb_api_key is set.
    wandb_run: str = None,  # Weights and Biases run name. Only applicable if wandb_api_key is set.
    wandb_entity: str = None,  # Weights and Biases entity name. Only applicable if wandb_api_key is set.
    wandb_sample_interval: int = 100,  # Step interval for sampling output images that are logged to W&B. Only applicable if wandb_api_key is set.
    wandb_sample_prompts: str = None,  # Newline-separated list of prompts to use when logging samples to W&B. Only applicable if wandb_api_key is set.
    wandb_save_interval: int = 100,  # Step interval for saving intermediate LoRA weights to W&B. Only applicable if wandb_api_key is set.
):
    clean_up()

    if not input_images:
        raise ValueError("input_images is required")

    layers_to_optimize = None
    if layers_to_optimize_regex:
        raise NotImplementedError("layers_to_optimize_regex is not implemented")

    sample_prompts = []
    if wandb_sample_prompts:
        sample_prompts = [p.strip() for p in wandb_sample_prompts.split("\n")]

    train_config = OrderedDict(
        {
            "job": "custom_job",
            "config": {
                "name": JOB_NAME,
                "process": [
                    {
                        "type": "custom_sd_trainer",
                        "training_folder": str(OUTPUT_DIR),
                        "device": "cuda:0",
                        "trigger_word": trigger_word,
                        "network": {
                            "type": "lora",
                            "linear": lora_rank,
                            "linear_alpha": lora_rank,
                        },
                        "save": {
                            "dtype": "float16",
                            "save_every": (
                                wandb_save_interval if wandb_api_key else steps + 1
                            ),
                            "max_step_saves_to_keep": 1,
                        },
                        "datasets": [
                            {
                                "folder_path": str(INPUT_DIR),
                                "caption_ext": "txt",
                                "caption_dropout_rate": caption_dropout_rate,
                                "shuffle_tokens": False,
                                # TODO: Do we need to cache to disk? It's faster not to.
                                "cache_latents_to_disk": cache_latents_to_disk,
                                "cache_latents": True,
                                "resolution": [
                                    int(res) for res in resolution.split(",")
                                ],
                            }
                        ],
                        "train": {
                            "batch_size": batch_size,
                            "steps": steps,
                            "gradient_accumulation_steps": 1,
                            "train_unet": True,
                            "train_text_encoder": False,
                            "content_or_style": "balanced",
                            "gradient_checkpointing": True,
                            "noise_scheduler": "flowmatch",
                            "optimizer": optimizer,
                            "lr": learning_rate,
                            "ema_config": {"use_ema": True, "ema_decay": 0.99},
                            "dtype": "bf16",
                        },
                        "model": {
                            "name_or_path": str(WEIGHTS_PATH),
                            "is_flux": True,
                            "quantize": True,
                        },
                        "sample": {
                            "sampler": "flowmatch",
                            "sample_every": (
                                wandb_sample_interval
                                if wandb_api_key and sample_prompts
                                else steps + 1
                            ),
                            "width": 1024,
                            "height": 1024,
                            "prompts": sample_prompts,
                            "neg": "",
                            "seed": 42,
                            "walk_seed": True,
                            "guidance_scale": 3.5,
                            "sample_steps": 28,
                        },
                    }
                ],
            },
            "meta": {"name": "[name]", "version": "1.0"},
        }
    )

    if layers_to_optimize:
        train_config["config"]["process"][0]["network"]["network_kwargs"] = {
            "only_if_contains": layers_to_optimize
        }

    # TODO: Implement wandb client

    extract_zip(input_images, INPUT_DIR)

    if not trigger_word:
        del train_config["config"]["process"][0]["trigger_word"]

    print("Starting train job")
    job = CustomJob(get_config(train_config, name=None), None)
    job.run()


def extract_zip(input_images: Path, input_dir: Path):
    if not is_zipfile(input_images):
        raise ValueError("input_images is not a zip file")

    input_dir.mkdir(parents=True, exist_ok=True)
    image_count = 0
    with ZipFile(input_images) as zip_ref:
        for file_info in zip_ref.infolist():
            if not file_info.filename.startswith(
                "__MACOSX/"
            ) and not file_info.filename.startswith("._"):
                zip_ref.extract(file_info, input_dir)
                image_count += 1

    print(f"Extracted {image_count} images from {input_images}")


def download_weights():
    if not WEIGHTS_PATH.exists():
        subprocess.check_output(
            [
                "wget",
                "-O",
                str(WEIGHTS_PATH.parent / "files.tar"),
                "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar",
            ]
        )
        subprocess.check_output(
            [
                "tar",
                "--no-same-owner",
                "-v",
                "-xf",
                str(WEIGHTS_PATH.parent / "files.tar"),
                "-C",
                str(WEIGHTS_PATH.parent),
            ]
        )
        Path(WEIGHTS_PATH.parent / "files.tar").unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["wandb_api_key"] = os.getenv("WANDB_API_KEY")

    train(**config)
