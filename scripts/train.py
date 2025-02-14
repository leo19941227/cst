import importlib
from cst.utils import initialize, get_trainer, find_last_ckpt


def import_cls(path):
    parts = path.split(".")
    module_path = ".".join(parts[:-1])
    class_name = parts[-1]
    module = importlib.import_module(module_path)
    CLS = getattr(module, class_name)
    return CLS


if __name__ == "__main__":
    conf = initialize()

    model_cls = import_cls(conf["target"])
    model = model_cls(**conf["model"])
    model.setup_data(**conf["data"])

    train_dataloader = model.get_dataloader("train")
    valid_dataloader = model.get_dataloader("valid")

    trainer = get_trainer(
        conf["expdir"],
        conf["trainer"],
        conf["save_steps"],
        conf["valid_metric"],
        conf["valid_higher_better"],
        conf.get("save_epoch", False),
    )

    last_ckpt = find_last_ckpt(conf["expdir"])
    trainer.fit(model, train_dataloader, valid_dataloader, ckpt_path=last_ckpt)
