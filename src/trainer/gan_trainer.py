import torch

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.misc import freeze_model


class GANTrainer(BaseTrainer):
    """
    GANTrainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        # DISCR

        freeze_model(self.model.gen, True)
        freeze_model(self.model.discr, False)

        with torch.no_grad():
            outputs = self.model(**batch)

        if self.is_train:
            self.optimizer["discr"].zero_grad()

        outputs.update(
            self.model.discriminate(batch["orig"].detach(), key_suff="_orig")
        )
        outputs.update(
            self.model.discriminate(outputs["recon"].detach(), key_suff="_recon")
        )

        batch.update(outputs)

        discr_losses = self.criterion["discr"](**batch)
        batch.update(discr_losses)

        if self.is_train:
            batch["discr_loss"].backward()
            self._clip_grad_norm(self.model.discr)
            self.optimizer["discr"].step()
            if self.lr_scheduler is not None and "discr" in self.lr_scheduler.keys():
                self.lr_scheduler["discr"].step()

        # GEN

        freeze_model(self.model.gen, False)
        freeze_model(self.model.discr, True)

        if self.is_train:
            self.optimizer["gen"].zero_grad()

        outputs.update(self.model(**batch))
        outputs.update(
            self.model.discriminate(batch["orig"].detach(), key_suff="_orig")
        )
        outputs.update(self.model.discriminate(outputs["recon"], key_suff="_recon"))

        batch.update(outputs)

        gen_losses = self.criterion["gen"](**batch)
        batch.update(gen_losses)

        if self.is_train:
            batch["gen_loss"].backward()
            self._clip_grad_norm(self.model.gen)
            self.optimizer["gen"].step()
            if self.lr_scheduler is not None and "gen" in self.lr_scheduler.keys():
                self.lr_scheduler["gen"].step()

        # update metrics for each loss (in case of multiple losses)
        batch["loss"] = batch["gen_loss"] + batch["discr_loss"]

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass
