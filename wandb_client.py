import wandb


def init(api_key: str, project: str, entity: str, config: dict, name: str):
    wandb.login(key=api_key)
    try:
        wandb.init(
            project=project,
            entity=entity,
            config=config,
            name=name,
        )
    except Exception as e:
        raise ValueError(f"Failed to log in to Weights & Biases: {e}")


def log_loss(loss_dict: dict, step: int | None):
    try:
        wandb.log(data=loss_dict, step=step)
    except Exception as e:
        print(f"Failed to log loss to Weights & Biases: {e}")


def finish():
    try:
        wandb.finish()
    except Exception:
        pass
