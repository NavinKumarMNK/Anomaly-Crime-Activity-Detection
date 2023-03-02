


class LSTMLogger(pl.Callback):
    def __init__(self, model:Decoder, data:CrimeActivityLSTMDataset) -> None:
        super(LSTMLogger, self).__init__()
        self.model = model
        self.data = data

    def on_train_start(self, trainer, pl_module):
        wandb.watch(self.model, log="all")

    def on_train_epoch_end(self, trainer, pl_module):
        wandb.watch(self.model, log="all")

    def on_test_end(self, trainer, pl_module):
        wandb.watch(self.model, log="all")

    def on_validation_epoch_end(self, trainer, pl_module):
        logits = pl_module(self.data.val_dataloader())
        preds = torch.argmax(logits, dim=1)
        print("Logging validation metrics")
        trainer.logger.experiment.log({
            "val_acc": torchmetrics.functional.accuracy(preds, self.data.val_y),
            "val_auc": torchmetrics.functional.auc(preds, self.data.val_y),
            "examples" : [wandb.Video(video, fps=4, format="mp4", caption= f"Pred: {pred} - Label: {label}")
                            for video, pred, label in zip(self.data.val_x, preds, self.data.val_y)],
            "global_step": trainer.global_step,          
            }, step=trainer.global_step)

if __name__ == '__main__': 
    trainer_params = utils.config_parse('LRCN_TRAIN')
    trainer = pl.Trainer(**trainer_params, #logger=logger,    
                        )
    model_params = utils.config_parse('LRCN_MODEL')
    model = Decoder(**model_params)
    
    trainer.fit(model)
    data = CrimeActivityLSTMDataset()
    trainer.test(datamodule=data)
    model.finalize()
