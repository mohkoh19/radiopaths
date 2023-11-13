from transformers import AutoModel
import torch.nn as nn
import torch
from torch.nn import BCEWithLogitsLoss
import pytorch_lightning as pl
from torchmetrics.functional import accuracy,f1_score
from util.logging import WandbLightningModule

class ClinooClassifier(WandbLightningModule):
    def __init__(
        self,
        pretrained="dmis-lab/biobert-v1.1",
        **kwarg,
    ):
        """ BioBERT classifier for clinical notes

        Args:
            pretrained (str, optional): Model name from Hugging Face. Defaults to "dmis-lab/biobert-v1.1".
        """
        super().__init__(**kwarg)
        self.save_hyperparameters(ignore=[*kwarg.keys()])

        self.backbone_model = AutoModel.from_pretrained(
            self.hparams.pretrained, output_hidden_states=True, output_attentions=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.backbone_model.pooler.dense.out_features,14),
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = WandbLightningModule.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("ClinoClassifier")
        parser.add_argument("--pretrained", type=str, default="dmis-lab/biobert-v1.1")
        return parent_parser

    def forward(self, x):
        if type(x) is dict and "clino" in x.keys():
            x = x["clino"]

        x = self.backbone_model(**x).pooler_output
        return self.classifier(x)
class ClinoClassifier(pl.LightningModule):
    def __init__(
        self,
        pretrained="dmis-lab/biobert-v1.1",
    ):
        """ BioBERT classifier for clinical notes

        Args:
            pretrained (str, optional): Model name from Hugging Face. Defaults to "dmis-lab/biobert-v1.1".
        """
        super(ClinoClassifier, self).__init__()
        self.save_hyperparameters()
        self.pretrained=pretrained
        self.backbone_model = AutoModel.from_pretrained(
            self.pretrained, output_hidden_states=True, output_attentions=True
        )
        self.num_classes=14

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.backbone_model.pooler.dense.out_features, self.num_classes),
        )

    def bceloss(self,x,labels):
      loss_fct = BCEWithLogitsLoss()
      return(loss_fct(x.view(-1, self.num_classes), labels.float()))

    def training_step(self,batch,idx):
      x,y=batch
      scores=self.forward(x)
      loss =self.bceloss(scores,y)
      self.log('train_loss', loss)
      return {"loss":loss,"scores":scores,"targets":y}


    def training_epoch_end(self, outputs):
      acc=[]
      f1=[]
      for output in outputs:
        scores=output["scores"]
        targets=output["targets"]
        acc_s=accuracy(scores,targets,task='multilabel',num_labels=14)
        f1_s=f1_score(scores,targets,task='multilabel',num_labels=14)
        f1.append(f1_s)
        acc.append(acc_s)
      final_acc=sum(acc)/len(acc)
      final_f1=sum(f1)/len(f1)
      self.log_dict({'accuracy':final_acc,"f1_score":final_f1})

    def validation_step(self, val_batch, batch_idx):
      x,y = val_batch
      scores = self.forward(x)
      loss=self.bceloss(scores,y)
      self.log('val_loss', loss)  

    def configure_optimizers(self):
      optimizer = torch.optim.AdamW(self.parameters(), lr=3e-5,weight_decay=0.01)
      scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5)
      return {"optimizer":optimizer,"lr_scheduler":{"scheduler":scheduler,"monitor":"val_loss","frequency":1}}

    def forward(self, x):
        if type(x) is dict and "clino" in x.keys():
          x = x["clino"]
        #for ftr in x.keys():
         # x[ftr]=x[ftr].to(torch.int64)
        x = self.backbone_model(**x).pooler_output
        return self.classifier(x)
        
