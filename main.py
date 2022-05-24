import os
import argparse
import pytorch_lightning as pl

from model import Model
import data

SAVE_DIR = "ADD SAVE DIR!!!"
PROJECT_NAME = "ADD WANDB PROJECT NAME!!!"
PROJECT_ENTITY = "ADD WANDB ENTITY!!!"


def train(args):
    pl.seed_everything(args.seed)

    save_name = args.model_name + "-trial-" + args.trial

    logger = pl.loggers.WandbLogger(project=PROJECT_NAME,
                                    entity=PROJECT_ENTITY,
                                    name=save_name)

    callbacks = [
        pl.callbacks.ModelCheckpoint(dirpath=os.path.join(SAVE_DIR, save_name),
                                     every_n_epochs=args.save_freq)
    ]

    model = Model()

    train_loader = data.get_train_loader()
    val_loader = data.get_val_loader()

    trainer = pl.Trainer(gpus=args.gpus,
                         check_val_every_n_epoch=args.eval_freq,
                         max_epochs=args.max_epochs,
                         logger=logger,
                         callbacks=callbacks)
    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)


def test(args):
    logger = pl.loggers.WandbLogger(project=PROJECT_NAME,
                                    entity=PROJECT_ENTITY,
                                    name=args.save_name)

    checkpoint = os.path.join(SAVE_DIR, args.checkpoint)

    model = Model.load_from_checkpoint(checkpoint)

    test_loader = data.get_test_loader()

    trainer = pl.Trainer(gpus=args.gpus, logger=logger)
    trainer.test(model, dataloaders=test_loader)


def main(args):
    if args.train:
        train(args)
    elif args.test:
        test(args)
    else:
        raise ValueError("Must either be in train or test mode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General args
    parser.add_argument("--gpus", nargs="+", default=None)

    sp = parser.add_subparsers(dest="action")

    # Training hparams
    train = sp.add_parser("train")
    train.add_argument("--model-name", type=str, required=True)
    train.add_argument("--trial", type=int, required=True)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--save-freq", type=int, default=1)
    train.add_argument("--eval-freq", type=int, default=1)
    train.add_argument("--resume-path", type=str, default=None)
    train.add_argument("--epochs", type=int, required=True)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=0.0)

    # Testing hparams
    test = sp.add_parser("test")
    test.add_argument("--save-name", type=str, required=True)
    test.add_argument("--checkpoint", type=str, required=True)

    args = parser.parse_args()

    main(args)