import itertools

import torch
from torchmetrics.functional import (
    accuracy,
    auc,
    auroc,
    average_precision,
    confusion_matrix,
    fbeta_score,
    precision,
    precision_recall_curve,
    recall,
    roc,
)

import wandb
from models.base_module import BaseModule


class WandbLightningModule(BaseModule):
    def __init__(self, threshold_tuning="f1", **kwarg):
        """ Class that extends the BaseModule by logging functionality using WandB

        Args:
            threshold_tuning (str, optional): Tuning method, "f1" or "roc". Defaults to "f1".
        """
        super().__init__(**kwarg)
        self.save_hyperparameters(ignore=[*kwarg.keys()])

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = BaseModule.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("Logging")
        parser.add_argument("--threshold_tuning", type=str, default="f1")
        return parent_parser

    def on_fit_start(self):
        super().on_fit_start()

        self.logger.experiment.define_metric("train/acc_pneumonia", summary="max")
        self.logger.experiment.define_metric("train/f1_pneumonia", summary="max")
        self.logger.experiment.define_metric("train/loss_epoch", summary="min")
        self.logger.experiment.define_metric("train/loss_step", summary="min")

        self.logger.experiment.define_metric("val/acc_pneumonia", summary="max")
        self.logger.experiment.define_metric("val/f1_pneumonia", summary="max")
        self.logger.experiment.define_metric("val/precision_pneumonia", summary="max")
        self.logger.experiment.define_metric("val/recall_pneumonia", summary="max")
        self.logger.experiment.define_metric("val/auroc_pneumonia", summary="max")
        self.logger.experiment.define_metric("val/auprc_pneumonia", summary="max")
        self.logger.experiment.define_metric("val/average_precision", summary="max")
        self.logger.experiment.define_metric("val/loss_epoch", summary="min")
        self.logger.experiment.define_metric("val/loss_step", summary="min")

        self.logger.experiment.define_metric("val/acc_macro", summary="max")
        self.logger.experiment.define_metric("val/f1_macro", summary="max")
        self.logger.experiment.define_metric("val/precision_macro", summary="max")
        self.logger.experiment.define_metric("val/recall_macro", summary="max")
        self.logger.experiment.define_metric("val/auroc_macro", summary="max")
        self.logger.experiment.define_metric("val/auprc_macro", summary="max")
        self.logger.experiment.define_metric("val/average_macro", summary="max")

    def on_test_start(self):
        super().on_test_start()

        self.logger.experiment.define_metric("test/acc_pneumonia", summary="max")
        self.logger.experiment.define_metric("test/f1_pneumonia", summary="max")
        self.logger.experiment.define_metric("test/precision_pneumonia", summary="max")
        self.logger.experiment.define_metric("test/recall_pneumonia", summary="max")
        self.logger.experiment.define_metric("test/auroc_pneumonia", summary="max")
        self.logger.experiment.define_metric("test/auprc_pneumonia", summary="max")
        self.logger.experiment.define_metric("test/average_precision", summary="max")
        self.logger.experiment.define_metric("test/loss_epoch", summary="min")
        self.logger.experiment.define_metric("test/loss_step", summary="min")

        self.logger.experiment.define_metric("test/acc_macro", summary="max")
        self.logger.experiment.define_metric("test/f1_macro", summary="max")
        self.logger.experiment.define_metric("test/precision_macro", summary="max")
        self.logger.experiment.define_metric("test/recall_macro", summary="max")
        self.logger.experiment.define_metric("test/auroc_macro", summary="max")
        self.logger.experiment.define_metric("test/auprc_macro", summary="max")
        self.logger.experiment.define_metric("test/average_macro", summary="max")

    def stack_outputs(self, outputs):
        """ Stacks all the output logits from an epoch for target disease.
            Transforms them to probabilities with sigmoid.
            Returns probabilities with ground truth.

        Args:
            outputs (list): List of dictionaries. Each containing logits and outputs.

        Returns:
            tuple: A list of probabilities and a list of their ground truth
        """
        logits = torch.cat([x["logits"] for x in outputs])
        probs = logits.sigmoid()
        ground_truth = torch.cat([x["y"] for x in outputs]).int()

        return (
            probs[:, self.target_idx],
            ground_truth[:, self.target_idx],
        )

    def optimal_threshold(self, probs, ground_truth):
        """ Returns the optimat threshold based on the chosen threshold tuning method.

        Args:
            probs (list): List of probabilities.
            ground_truth (list): List of ground truths.

        Returns:
            float: Optimal threshold
        """
        tune = self.hparams.threshold_tuning
        if tune.startswith("f"):
            return self._fbeta_threshold(probs, ground_truth, tune)
        elif tune == "roc":
            fpr, tpr, roc_th = roc(probs, ground_truth, pos_label=1)
            return self._roc_threshold(fpr, tpr, roc_th)

    def _fbeta_threshold(self, probs, ground_truth, tune):
        # Calculate the f-beta-scores
        beta = float(tune[1:])
        thresholds = torch.arange(0, 1, 0.001, device=self.device)
        fscores = torch.tensor(
            [fbeta_score(probs, ground_truth, beta, threshold=th) for th in thresholds],
            device=self.device,
        )

        # Find the optimal threshold
        index = torch.argmax(fscores)

        return thresholds[index]

    def _roc_threshold(self, fpr, tpr, th):
        # Calculate the G-mean
        gmean = torch.sqrt(tpr * (1 - fpr))

        # Find the optimal threshold
        index = torch.argmax(gmean)

        return th[index] 

    def plot_roc_curve(self, fpr, tpr):
        """ Plots ROC curve in WandB

        Args:
            fpr (list): False Positive Rates.
            tpr (list): True Positive Rates.

        Returns:
            plot: WandB plot
        """
        fpr = fpr.detach().cpu().tolist()
        tpr = tpr.detach().cpu().tolist()

        total = len(fpr)
        th_length = min(1000, total)
        step_size = total // th_length
        idx_list = []
        for i in range(th_length):
            if i == 0:
                idx_list.append(0)
            elif i == th_length - 1:
                idx_list.append(total - 1)
            else:
                idx_list.append(i * step_size)

        data = [["Pneumonia", fpr[i], tpr[i]] for i in idx_list]

        table = wandb.Table(columns=["class", "fpr", "tpr"], data=data)
        title = "ROC"
        return wandb.plot_table(
            "wandb/area-under-curve/v0",
            table,
            {"x": "fpr", "y": "tpr", "class": "class"},
            {
                "title": title,
                "x-axis-title": "False positive rate",
                "y-axis-title": "True positive rate",
            },
        )

    def plot_pr_curve(self, precisions, recalls):
        """ Plots PR curve in WandB

        Args:
            precisions (list): Precisions.
            recalls (list): Recalls.

        Returns:
            plot: WandB plot
        """
        precisions = precisions.detach().cpu().tolist()
        recalls = recalls.detach().cpu().tolist()

        total = len(precisions)
        th_length = min(1000, total)
        step_size = total // th_length
        idx_list = []
        for i in range(th_length):
            if i == 0:
                idx_list.append(0)
            elif i == th_length - 1:
                idx_list.append(total - 1)
            else:
                idx_list.append(i * step_size)

        data = [["Pneumonia", precisions[i], recalls[i]] for i in idx_list]

        table = wandb.Table(columns=["class", "precision", "recall"], data=data)
        title = "Precision v. Recall"
        return wandb.plot_table(
            "wandb/area-under-curve/v0",
            table,
            {"x": "recall", "y": "precision", "class": "class"},
            {
                "title": title,
                "x-axis-title": "Recall",
                "y-axis-title": "Precision",
            },
        )

    def plot_confusion_matrix(self, conf_mat):
        """ Plots Confusion Matrix.

        Args:
            conf_mat (Tensor): Confusion Matrix

        Returns:
            plot: WandB plot
        """
        conf_mat = conf_mat[0]

        conf_mat = conf_mat.detach().cpu().numpy()

        label_names = ["negative", "positive"]

        it = itertools.product([0, 1], [0, 1])

        data = [[label_names[i], label_names[j], conf_mat[(i, j)]] for (i, j) in it]

        fields = {
            "Actual": "Actual",
            "Predicted": "Predicted",
            "nPredictions": "nPredictions",
        }
        title = "Confusion Matrix"
        return wandb.plot_table(
            "wandb/confusion_matrix/v1",
            wandb.Table(columns=["Actual", "Predicted", "nPredictions"], data=data),
            fields,
            {"title": title},
        )

    def training_epoch_end(self, outputs):
        """ Logging at the end of each training epoch.

        Args:
            outputs (list): List of dictionaries. Each containing logits and outputs.
        """
        probs, ground_truth = self.stack_outputs(outputs)

        # Optimal Threshold
        opt_th = self.optimal_threshold(probs, ground_truth)

        # Threshold Metrics
        acc = accuracy(probs, ground_truth, threshold=opt_th)
        f1 = fbeta_score(probs, ground_truth, threshold=opt_th, beta=1.0)

        self.log_dict(
            {
                "train/acc_pneumonia": acc,
                "train/f1_pneumonia": f1,
            }
        )

    def validation_epoch_end(self, outputs):
        """ Logging at the end of each validation epoch.

        Args:
            outputs (list): List of dictionaries. Each containing logits and outputs.
        """
        probs, ground_truth = self.stack_outputs(outputs)

        # Ranking Metrics
        precs, recs, _ = precision_recall_curve(probs, ground_truth, pos_label=1)
        auprc = auc(recs, precs)

        fpr, tpr, _ = roc(probs, ground_truth, pos_label=1)
        auroc_pneumonia = auc(fpr, tpr)

        avg_prec = average_precision(probs, ground_truth)

        # Optimal Threshold
        opt_th = self.optimal_threshold(probs, ground_truth)

        # Threshold Metrics
        acc = accuracy(probs, ground_truth, threshold=opt_th)
        f1 = fbeta_score(probs, ground_truth, threshold=opt_th, beta=1.0)
        prec = precision(probs, ground_truth, threshold=opt_th)
        rec = recall(probs, ground_truth, threshold=opt_th)

        # Confusion Matrix
        conf_mat = confusion_matrix(probs, ground_truth, num_classes=1, multilabel=True, threshold=opt_th)

        # Macro Metrics across all diseases
        probs, ground_truth = self.stack_outputs(outputs, target_only=False)
        acc_macro = accuracy(probs, ground_truth, threshold=opt_th, average="macro", num_classes=self.num_classes)
        f1_macro = fbeta_score(
            probs, ground_truth, threshold=opt_th, beta=1.0, average="macro", num_classes=self.num_classes
        )
        prec_macro = precision(probs, ground_truth, threshold=opt_th, average="macro", num_classes=self.num_classes)
        rec_macro = recall(probs, ground_truth, threshold=opt_th, average="macro", num_classes=self.num_classes)
        avg_prec_macro = average_precision(probs, ground_truth, average="macro", num_classes=self.num_classes)
        auroc_macro = auroc(probs, ground_truth, average="macro", num_classes=self.num_classes)

        self.log_dict(
            {
                "val/acc_pneumonia": acc,
                "val/f1_pneumonia": f1,
                "val/precision_pneumonia": prec,
                "val/recall_pneumonia": rec,
                "val/auroc_pneumonia": auroc_pneumonia,
                "val/auprc_pneumonia": auprc,
                "val/average_precision": avg_prec,
                "val/optimal_threshold": opt_th,
                "val/acc_macro": acc_macro,
                "val/f1_macro": f1_macro,
                "val/precision_macro": prec_macro,
                "val/recall_macro": rec_macro,
                "val/auroc_macro": auroc_macro,
                "val/average_precision_macro": avg_prec_macro,
            },
        )

        self.logger.experiment.log(
            {
                "val/conf_mat_pneumonia": self.plot_confusion_matrix(conf_mat),
            }
        )

    def test_epoch_end(self, outputs):
        """ Logging at the end of each test epoch.

        Args:
            outputs (list): List of dictionaries. Each containing logits and outputs.
        """
        probs, ground_truth = self.stack_outputs(outputs)

        # Ranking Metrics
        precs, recs, _ = precision_recall_curve(probs, ground_truth, pos_label=1)
        auprc = auc(recs, precs)

        fpr, tpr, _ = roc(probs, ground_truth, pos_label=1)
        auroc_pneumonia = auc(fpr, tpr)

        avg_prec = average_precision(probs, ground_truth)

        # Optimal Threshold
        opt_th = self.optimal_threshold(probs, ground_truth)

        # Threshold Metrics
        acc = accuracy(probs, ground_truth, threshold=opt_th)
        f1 = fbeta_score(probs, ground_truth, threshold=opt_th, beta=1.0)
        prec = precision(probs, ground_truth, threshold=opt_th)
        rec = recall(probs, ground_truth, threshold=opt_th)

        # Macro Metrics across all diseases
        probs, ground_truth = self.stack_outputs(outputs, target_only=False)
        acc_macro = accuracy(probs, ground_truth, threshold=opt_th, average="macro", num_classes=self.num_classes)
        f1_macro = fbeta_score(
            probs, ground_truth, threshold=opt_th, beta=1.0, average="macro", num_classes=self.num_classes
        )
        prec_macro = precision(probs, ground_truth, threshold=opt_th, average="macro", num_classes=self.num_classes)
        rec_macro = recall(probs, ground_truth, threshold=opt_th, average="macro", num_classes=self.num_classes)
        avg_prec_macro = average_precision(probs, ground_truth, average="macro", num_classes=self.num_classes)
        auroc_macro = auroc(probs, ground_truth, average="macro", num_classes=self.num_classes)

        # Confusion Matrix
        conf_mat = confusion_matrix(probs, ground_truth, num_classes=1, multilabel=True, threshold=opt_th)

        self.log_dict(
            {
                "test/acc_pneumonia": acc,
                "test/f1_pneumonia": f1,
                "test/precision_pneumonia": prec,
                "test/recall_pneumonia": rec,
                "test/auroc_pneumonia": auroc_pneumonia,
                "test/auprc_pneumonia": auprc,
                "test/average_precision": avg_prec,
                "test/optimal_threshold": opt_th,
                "test/acc_macro": acc_macro,
                "test/f1_macro": f1_macro,
                "test/precision_macro": prec_macro,
                "test/recall_macro": rec_macro,
                "test/auroc_macro": auroc_macro,
                "test/average_precision_macro": avg_prec_macro,
            },
        )

        roc_plot = self.plot_roc_curve(fpr, tpr)
        pr_plot = self.plot_pr_curve(precs, recs)
        conf_mat_plot = self.plot_confusion_matrix(conf_mat)

        self.logger.experiment.log(
            {
                "test/roc_curve_pneumonia": roc_plot,
                "test/pr_curve_pneumonia": pr_plot,
                "test/conf_mat_pneumonia": conf_mat_plot,
            },
            commit=True,
        )
