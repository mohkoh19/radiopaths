# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

from torchmetrics.functional.classification import multilabel_auroc,multilabel_average_precision,multilabel_precision_recall_curve
from sklearn.metrics import average_precision_score
import models.configs as configs
from models.attention import Attention
from models.embed import Embeddings 
from models.mlp import Mlp
from models.block import Block
from models.clino import ClinoClassifier
from models.encoder import Encoder
from torchmetrics.functional import accuracy,f1_score
import pytorch_lightning as pl
import pdb

logger = logging.getLogger(__name__)

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class BinaryFocalLossWithLogits(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=1, pos_weight=None):
        """ Focal loss as in https://arxiv.org/abs/1708.02002

        Args:
            weight (float, optional): Class weight (alpha parameter in paper). Defaults to None.
            gamma (int, optional): _description_. Defaults to 2.
            pos_weight (float, optional): Per-class skew (beta parameter in paper). Defaults to None.
        """
        super().__init__()
        self.gamma = gamma

        # weight parameter will act as the alpha parameter to balance class weights
        self.weight = weight

        self.pos_weight = pos_weight

    def forward(self, input, target):
        bce_loss = F.binary_cross_entropy_with_logits(
            input, target, reduction="none", weight=self.weight, pos_weight=self.pos_weight
        )
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma * bce_loss).mean()

        return focal_loss

class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids, cc=None, lab=None, sex=None, age=None):
        embedding_output, cc, lab, sex, age = self.embeddings(input_ids, cc, lab, sex, age)    
        text = torch.cat((cc, lab, sex, age), 1)
        encoded, attn_weights = self.encoder(embedding_output, text)
        return encoded, attn_weights


class IRENE(pl.LightningModule):
    def __init__(self, config, size_train,img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(IRENE, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.tk_lim = config.cc_len
        self.num_lab = config.lab_len
       
        self.size_train=size_train
        self.clino_backbone=ClinoClassifier.load_from_checkpoint('/content/clino_biobert_final-epoch=2-val_loss=0.32.ckpt',strict=False)
       
        self.clino_backbone.classifier = nn.Identity()
        self.clino_backbone.freeze()
        self.loss = BinaryFocalLossWithLogits()
        self.pneumonia=[]
        self.atel=[]
        self.cardi=[]
        self.consol=[]
        self.edema=[]
        self.enlarged=[]
        self.fracture=[]
        self.lung_lesion=[]
        self.lung_opac=[]
        self.pleural_eff=[]
        self.pleural_other=[]
        self.pneumothorax=[]

        self.pneumonia2=[]
        self.atel2=[]
        self.cardi2=[]
        self.consol2=[]
        self.edema2=[]
        self.enlarged2=[]
        self.fracture2=[]
        self.lung_lesion2=[]
        self.lung_opac2=[]
        self.pleural_eff2=[]
        self.pleural_other2=[]
        self.pneumothorax2=[]



        self.transformer = Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, cc=None, lab=None, sex=None, age=None):
        x, attn_weights = self.transformer(x, cc, lab, sex, age)
        logits = self.head(torch.mean(x, dim=1))
      
        return logits

        #if labels is not None:
         #   loss_fct = BCEWithLogitsLoss()
            #print("labels",labels)
            #print("logits",logits.shape)
          
            #print("inter",logits.view(-1, self.num_classes).shape)
          #  loss = loss_fct(logits.view(-1, self.num_classes), labels.float())
            #print("loss",loss)
           # return loss
        #else:
            #return logits, attn_weights, torch.mean(x, dim=1)

    def bceloss(self,x,labels):
      loss_fct = BCEWithLogitsLoss()
      return(loss_fct(x.view(-1, self.num_classes), labels.float()))

    def training_step(self,batch,idx):
        features,y,index=batch
        print(self.size_train)
        for a in range(y.shape[0]):
          z=torch.count_nonzero(y[a])
          if z==0:
            print("aq",y[a])


        x=features['image']
       
        text=self.clino_backbone(features)
  
        #text=features['clino']
 
        #text = text.view(-1, self.tk_lim,354).to(torch.float32)

        lab=torch.unsqueeze(features['clico'],2)

        sex=features['demog'][:,0]
        sex=sex.view(-1,1,1)
        age=features['demog'][:,1]
        age=age.view(-1,1,1)
  
      
        scores=self.forward(x,text,lab,sex,age)
        #loss_fn = self.loss
        #loss = loss_fn(scores,y)
        
        loss =self.bceloss(scores,y)
        
        self.log('train_loss', loss,on_step=True, on_epoch=True)
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
        features,y,index = val_batch
        x=features['image']
        #text=features['clino']['input_ids']
        #text = text.view(-1, self.tk_lim,354).to(torch.float32)
        text=self.clino_backbone(features)
        
        lab=lab=torch.unsqueeze(features['clico'],2)
        sex=features['demog'][:,0]
        sex=sex.view(-1,1,1)
        age=features['demog'][:,1]
        age=age.view(-1,1,1)       
        scores=self.forward(x,text,lab,sex,age)
        #loss_fn = self.loss
        #loss = loss_fn(scores,y)
        loss =self.bceloss(scores,y)
        self.log('val_loss', loss,on_step=True,on_epoch=True) 

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5,weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=5)
        return {"optimizer":optimizer,"lr_scheduler":{"scheduler":scheduler,"monitor":"val_loss","frequency":1}}


    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            # print(posemb.size(), posemb_new.size())
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


CONFIGS = {
    'IRENE': configs.get_IRENE_config(),
}
