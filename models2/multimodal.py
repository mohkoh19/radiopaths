import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from util.util import semi_flag
from util.logging import WandbLightningModule
from self_attention_cv import RelPosEmb2D
from util.vision import prepare_backbone
from util.vision import AttentionConv
import torch.nn.functional as F
from models.clino import ClinoClassifier
from .unimodal import CXRClassifier, ClicoClassifier
from .transformer_encoder import CrossAttentionBlock, PositionalEncoding
from util.vision import AttentionConv
from torch.nn import BCEWithLogitsLoss
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from torchmetrics.functional import accuracy,f1_score
from util._curriculum_clustering import CurriculumClustering
import torch.optim as optim
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
class CustomCyclicLR(optim.lr_scheduler._LRScheduler):
    ##custom learning rate scheduler to change learning rate cyclicly for o2u Net
    def __init__(self, optimizer, r1, r2, c, num_cycles, last_epoch=-1):
        self.r1 = r1
        self.r2 = r2
        self.c = c
        self.num_cycles = num_cycles
        super(CustomCyclicLR, self).__init__(optimizer, last_epoch)
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]


    def get_lr(self):
        cycle_progress = (self.last_epoch - 1) % self.c + 1
        #s = (epoch %self.c + 1) /self.c
        s = (1 + cycle_progress - 1) / self.c
        lr = (1 - s) * self.r2 + s * self.r1
        if cycle_progress == self.c:  #Reset to the initial learning rate after completing one cycle
          lr = self.r2
        return [lr for _ in self.base_lrs]

def save_lists(informative, confident, noisy, file_name='lists.pkl'):
  data = {'informative': informative, 'confident': confident, 'noisy': noisy}
  with open(file_name, 'wb') as file:
    pickle.dump(data, file)
class CXR2Classifier(WandbLightningModule):
    def __init__(
        self,
        num_heads=8,
        dropout=0.1,
        dim_feedforward=2048,
        pretrained=True,
        backbone="efficientnet b0",
        cxr_pretrained=None,
        layer_1=0,
        **kwarg,
    ):
        """ Multi-View CXR classifier.
            Feature extractor is an EfficientNet B0 CNN pre-trained on ImageNet
            Either fine-tuned on same dataset & frozen weights or no fine-tuning and no frozen weights.
            Feature output are local features of shape 1280 x 7 x 7
            The high dimensionality might be prhibitive to apply attention
            Instead: Use the penultimate convolutional layer that outputs 320 X 7 X 7 features
            The last convolutional layer in EfficientNet is a 1x1 Conv2D layer, thus we just append it
            In this way we "inject" attention between the CNN feature maps.

        Args:
            num_heads (int, optional): Number of heads for MHA. Defaults to 8.
            dropout (float, optional): Dropout rate in Encoder. Defaults to 0.1.
            dim_feedforward (int, optional): Size of hidden layer in Encoder FFN. Defaults to 2048.
            pretrained (bool, optional): Uses model pre-trained on ImageNet if enabled. Defaults to True.
            backbone (str, optional): _description_. Defaults to "efficientnet b0".
            cxr_pretrained (str, optional): Path to other pre-trained model. Defaults to None.
            layer_1 (int, optional): If not 0, adds a linear layer with the provided size and ReLU activation before the final classification layer. Defaults to 0.
        """
        super().__init__(**kwarg)
        self.save_hyperparameters(ignore=[*kwarg.keys()])

        # Initialize feature extractor
        if self.hparams.cxr_pretrained is not None:
            cxrc = CXRClassifier.load_from_checkpoint(self.hparams.cxr_pretrained)
            cxrc.freeze()
            backbone = cxrc.backbone_model
        else:
            backbone, _ = prepare_backbone(
                self.hparams.backbone,
                self.hparams.pretrained,
                0,
                self.num_classes,
            )

        # Replace Conv before and after cross attention with AttConv
        # Fix number of intermediate features and output features
        conv2 = backbone.features[-2][0].block[1][0]
        self.num_features = backbone.features[-2][0].block[3][0].out_channels
        backbone.features[-2][0].block[1][0] = AttentionConv(
            in_channels=conv2.in_channels,
            out_channels=conv2.out_channels,
            kernel_size=conv2.kernel_size[0],
            stride=conv2.stride[0],
            padding=conv2.padding[0],
            groups=self.hparams.num_heads,
        )

        conv2 = backbone.features[-1][0]
        self.out_features = conv2.out_channels

        self.cxr_xtr = backbone.features[:-1]
        self.upsample = nn.Sequential(backbone.features[-1], nn.AdaptiveAvgPool2d(1), nn.Flatten(1))

        # Unfreeze parameters of last feature layer and upsampling layer if pre-trained
        if self.hparams.cxr_pretrained is not None:
            for param in self.cxr_xtr[-1].parameters():
                param.requires_grad = True
            for param in self.upsample.parameters():
                param.requires_grad = True

        self.seq_length = 49
        self.fmap_size = (7, 7)

        # Cross-Attention is applied in 2 stages like in Liu et al., 2021
        ## We need to mask out all instances in the batch without lateral images
        self.cross_attention = CrossAttentionBlock(
            dim=self.num_features,
            heads=self.hparams.num_heads,
            dim_linear_block=self.hparams.dim_feedforward,
            dropout=self.hparams.dropout,
            activation=nn.SiLU,
            pe=RelPosEmb2D(self.fmap_size, dim_head=(self.num_features // self.hparams.num_heads)),
        )

        self.num_features = self.out_features

        # Classification of global representation
        layers = []

        ## optional intermediate layer
        if self.hparams.layer_1 != 0:
            layers.extend(
                [
                    nn.Linear(self.out_features, self.hparams.layer_1),
                    nn.ReLU(),
                ]
            )
            self.out_features = self.hparams.layer_1

        ## classification layer
        layers.append(nn.Linear(self.out_features, self.num_classes))

        self.classifier = nn.Sequential(*layers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = WandbLightningModule.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("CXR2Classifier")
        parser.add_argument("--pretrained", type=semi_flag, nargs="?", const=True, default=False)
        parser.add_argument("--backbone", type=str, default="efficientnet b0")
        parser.add_argument("--cxr_pretrained", type=str, default=None)
        parser.add_argument("--layer_1", type=int, default=0)
        parser.add_argument("--num_heads", type=int, default=8)
        parser.add_argument("--dim_feedforward", type=int, default=256)
        parser.add_argument("--dropout", type=float, default=0.1)
        return parent_parser

    # Attention map is 1 where lateral image is missing
    def _build_attention_mask(self, x):
        lateral = x["image"]["lateral"]

        mask = torch.zeros(
            (
                lateral.shape[0],
                self.hparams.num_heads,
                self.seq_length,
                self.seq_length,
            ),
            device=self.device,
        )

        missing = (lateral.isnan().sum((1, 2, 3)) != 0).nonzero().flatten().tolist()

        mask[missing] += 1

        return mask.bool()

    def _apply_on_non_missing(self, laterals, non_missing, func):

        lateral_features = func(laterals[non_missing])

        result = torch.zeros(
            (
                laterals.shape[0],
                lateral_features.shape[1],
                lateral_features.shape[2],
                lateral_features.shape[3],
            ),
            device=self.device,
            dtype=lateral_features.dtype,
        )

        result[non_missing] = lateral_features

        return result

    def _extract_features(self, x):
        # Extract frontal features
        frontal_features = self.cxr_xtr(x["image"]["frontal"])

        # Extract lateral features if at least one lateral exists
        laterals = x["image"]["lateral"]
        non_missing = (laterals.isnan().sum((1, 2, 3)) == 0).nonzero().flatten()

        if not non_missing.numel():
            lateral_features = torch.zeros_like(
                frontal_features,
                device=self.device,
                dtype=torch.half,
            )
        else:
            lateral_features = self._apply_on_non_missing(laterals, non_missing, self.cxr_xtr)

        # Sequentialize feature maps
        frontal_features = frontal_features.flatten(2, 3).transpose(1, 2)
        lateral_features = lateral_features.flatten(2, 3).transpose(1, 2)

        return frontal_features, lateral_features

    def forward(self, x):
        # Build attention mask based on missing laterals
        attn_mask = self._build_attention_mask(x)

        # Extract feature maps from CNN and sequentialize them
        frontal_features, lateral_features = self._extract_features(x)

        # Apply cross-attention
        frontal_features = self.cross_attention(X=frontal_features, Y=lateral_features, mask1=attn_mask)

        # Upsample & Pool
        frontal_features = self.upsample(frontal_features.transpose(1, 2).unflatten(2, self.fmap_size))

        # Classify global representation
        out = self.classifier(frontal_features)
        return out


class Early2Fusion(WandbLightningModule):
    def __init__(
        self,
        cxr_pretrained,
        clino_pretrained,
        clico_pretrained,
        cxr_type="single",
        layer_reduce_img=0,
        layer_reduce_clino=0,
        layer_1=1024,
        layer_2=512,
        dropout=0.1,
        **kwarg,
    ):
        """  Our Early Fusion Model using covariates and notes

        Args:
            cxr_pretrained (str): Path to pre-trained CXR-classifier.
            clino_pretrained (str): Path to pre-trained BioBERT model for clinical notes.
            clico_pretrained (str): Path to pre-trained FFN for clinical covariates.
            cxr_type (str, optional): Set to "multi" for multi-view and otherwise to "single". Defaults to "single".
            layer_reduce_img (int, optional): Optional linear layer of provided size with ReLU activation that reduced the CXR feature size. Defaults to 0.
            layer_reduce_clino (int, optional): Optional linear layer of provided size with ReLU activation that reduced the clinical notes feature size. Defaults to 0.
            layer_1 (int, optional): If not set to 0, creates a linear layer of provided size with ReLU activation before the classification layer. Defaults to 1024.
            layer_2 (int, optional): If not set to 0, creates another linear layer of provided size with ReLU activation before the classification layer. Defaults to 512.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        self.save_hyperparameters(ignore=[*kwarg.keys()])
        super().__init__(**kwarg)

        # Load pretrained CXRClassifier and "drop" last layer
        if self.hparams.cxr_type == "single":
            self.cxr_backbone = CXRClassifier.load_from_checkpoint(self.hparams.cxr_pretrained)
            self.cxr_backbone.backbone_model.classifier = nn.Identity()
        elif self.hparams.cxr_type == "multi":
            self.cxr_backbone = CXR2Classifier.load_from_checkpoint(self.hparams.cxr_pretrained)
            self.cxr_backbone.classifier = nn.Identity()

        self.num_img_features = self.cxr_backbone.num_features
        self.cxr_backbone.freeze()

        # Determine if image features should be reduced before fusion
        if self.hparams.layer_reduce_img != 0:
            self.layer_reduce_img = nn.Sequential(
                nn.Linear(self.num_img_features, self.hparams.layer_reduce_img),
                nn.Dropout(self.hparams.dropout),
                nn.ReLU(inplace=True),
            )
            self.num_img_features = self.hparams.layer_reduce_img
        else:
            self.layer_reduce_img = nn.Identity()

        # Load pretrained ClinoClassifier and "drop" last layer
        # TODO: retrain clino for new implementation with num_classes and target_idx...
        self.clino_backbone = ClinoClassifier.load_from_checkpoint(
            self.hparams.clino_pretrained, num_classes=14, target_idx=0
        )
        self.clino_backbone.classifier = nn.Identity()
        self.clino_backbone.freeze()

        # Determine if pooler output features should be reduced before fusion
        self.num_clino_features = self.clino_backbone.backbone_model.pooler.dense.out_features
        if self.hparams.layer_reduce_clino != 0:
            self.layer_reduce_clino = nn.Sequential(
                nn.Linear(self.num_clino_features, self.hparams.layer_reduce_clino),
                nn.Dropout(self.hparams.dropout),
                nn.ReLU(inplace=True),
            )
            self.num_clino_features = self.hparams.layer_reduce_clino
        else:
            self.layer_reduce_clino = nn.Identity()

        # Use pretrained Clico Classifier
        self.clico_backbone = ClicoClassifier.load_from_checkpoint(self.hparams.clico_pretrained)
        self.clico_backbone.classifier = nn.Identity()
        self.clico_backbone.freeze()
        self.num_clico_features = self.clico_backbone.num_features

        self.num_fusion_features = self.num_img_features + self.num_clico_features + self.num_clino_features

        # Determine how fusion features should be reduced before classification
        if self.hparams.layer_1 != 0:
            self.layer_1 = nn.Sequential(
                nn.Linear(self.num_fusion_features, self.hparams.layer_1),
                nn.Dropout(self.hparams.dropout),
                nn.ReLU(inplace=True),
            )
            self.num_fusion_features = self.hparams.layer_1
        else:
            self.layer_1 = nn.Identity()

        if self.hparams.layer_2 != 0:
            self.layer_2 = nn.Sequential(
                nn.Linear(self.num_fusion_features, self.hparams.layer_2),
                nn.Dropout(self.hparams.dropout),
                nn.ReLU(inplace=True),
            )
            self.num_fusion_features = self.hparams.layer_2
        else:
            self.layer_2 = nn.Identity()

        self.classifier = nn.Linear(self.num_fusion_features, self.num_classes)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = WandbLightningModule.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("Early2Fusion")
        parser.add_argument("--cxr_pretrained", type=str, required=True)
        parser.add_argument("--clino_pretrained", type=str, required=True)
        parser.add_argument("--clico_pretrained", type=str, required=True)
        parser.add_argument("--cxr_type", type=str, default="single")
        parser.add_argument("--layer_reduce_img", type=int, default=0)
        parser.add_argument("--layer_reduce_clino", type=int, default=0)
        parser.add_argument("--layer_1", type=int, default=1024)
        parser.add_argument("--layer_2", type=int, default=512)
        parser.add_argument("--dropout", type=float, default=0.1)
        return parent_parser

    def forward(self, x):
        img_features = self.cxr_backbone(x)
        clico_features = self.clico_backbone(x)
        clino_features = self.clino_backbone(x)

        # Reduce image features
        img_features = self.layer_reduce_img(img_features)

        # Reduce clino features
        clino_features = self.layer_reduce_clino(clino_features)

        # Concatenate reduced image features with clinical covariates and notes
        x_fusion = torch.cat((img_features, clico_features, clino_features), dim=1).float()

        # Further reduce through layer_1 and layer_2
        x_fusion = self.layer_1(x_fusion)
        x_fusion = self.layer_2(x_fusion)

        # Classify multimodal features
        y_hat_logits = self.classifier(x_fusion)

        return y_hat_logits


class Radiopaths(pl.LightningModule):
    def __init__(
        self,
        size_dataset,
        size_unlabelled,
        unlabelled_loader,
        labelled_loader,
        dim=512,
        num_heads=4,
        dropout=0.1,
        layer_1=0,
        **kwarg,
    ):
        
        self.save_hyperparameters(ignore=[*kwarg.keys()])
        super().__init__(**kwarg)
        self.clino_backbone=ClinoClassifier.load_from_checkpoint('/home/onur/clino_biobert_final-epoch=2-val_loss=0.32.ckpt')
        self.clico_backbone=ClicoClassifier.load_from_checkpoint('/home/onur/clico_ffn_final-epoch=6-val_loss=0.34.ckpt')
        self.cxr_backbone=CXRClassifier.load_from_checkpoint('/home/onur/cxr_effnet_b0_frontal_only_final-epoch=8-val_loss=0.30.ckpt')
        self.size_dataset=size_dataset
        self.size_unlabelled=size_unlabelled
        self.unlabelled_loader=unlabelled_loader
        self.labelled_loader=labelled_loader
        self.num_classes=14
        #self.moving_loss_dic=torch.zeros(self.size_dataset)
        #self.moving_entropy_dic=torch.zeros(self.size_unlabelled)
        self.moving_loss=[]  ##used for accumulating loss(
        self.moving_entropy=[] ##used for accumulating entropy
        self.cxr_backbone.backbone_model.classifier=nn.Identity()
        self.num_img_features = self.cxr_backbone.num_features
        #self.cxr_backbone.freeze()
        self.cxr_proj = nn.Linear(self.num_img_features, self.hparams.dim, bias=False)

        # Load pretrained ClinoClassifier and "drop" last layer
        # TODO: retrain clino for new implementation with num_classes and target_idx...
        #self.clino_backbone = ClinoClassifier.load_from_checkpoint(
         #   self.hparams.clino_pretrained, num_classes=14, target_idx=0
        #)
        self.clino_backbone.classifier = nn.Identity()
        self.num_clino_features = self.clino_backbone.backbone_model.pooler.dense.out_features
        self.clino_backbone.freeze()
        self.clino_proj = nn.Linear(self.num_clino_features, self.hparams.dim, bias=False)

        # Use pretrained Clico Classifier
        #self.clico_backbone = ClicoClassifier.load_from_checkpoint(self.hparams.clico_pretrained)
        self.clico_backbone.classifier = nn.Identity()
        self.clico_backbone.freeze()
        self.num_clico_features = self.clico_backbone.num_features
        self.clico_proj = nn.Linear(self.num_clico_features, self.hparams.dim, bias=False)
        self.r1=0.0001  #minimum learning rate
        self.r2=0.01  #maximum learning rate
        self.c=12   #number of epochs in each cyclical round
        self.num_cycles=3   #number of cyclical rounds
        self.pe = PositionalEncoding(self.hparams.dim, 3)

        self.mha = nn.MultiheadAttention(
            embed_dim=self.hparams.dim, dropout=self.hparams.dropout, num_heads=self.hparams.num_heads, batch_first=True
        )

        # Classification of global representation
        layers = []
        self.out_features = self.hparams.dim  # * 3

        ## optional intermediate layer
        if self.hparams.layer_1 != 0:
            layers.extend(
                [
                    nn.Linear(self.out_features, self.hparams.layer_1),
                    nn.ReLU(),
                ]
            )
            self.out_features = self.hparams.layer_1

        ## classification layer
        layers.append(nn.Linear(self.out_features, self.num_classes))

        self.classifier = nn.Sequential(*layers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = WandbLightningModule.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("Early2Fusion")
        parser.add_argument("--cxr_pretrained", type=str, required=True)
        parser.add_argument("--clino_pretrained", type=str, required=True)
        parser.add_argument("--clico_pretrained", type=str, required=True)
        parser.add_argument("--cxr_type", type=str, default="single")
        parser.add_argument("--dim", type=int, default=512)
        parser.add_argument("--num_heads", type=int, default=512)
        parser.add_argument("--layer_1", type=int, default=1024)
        parser.add_argument("--dropout", type=float, default=0.1)
        return parent_parser

    def _build_attention_mask(self, x):
        clico = x["clico"]
        clino = x["clino"]

        l1 = (clico.sum(1) == 0).nonzero().flatten().tolist()
        l2 = (clino["input_ids"].sum(1) == 0).nonzero().flatten().tolist()

        m0 = torch.zeros(
            (clico.shape[0], 3, 3),
            dtype=torch.int16,
            device=self.device,
        )

        m0[l1, 1, :] += 1
        m0[l2, 2, :] += 1

        m0 = m0.transpose(1, 2).unsqueeze(1).repeat(1, self.hparams.num_heads, 1, 1).bool()

        return m0

    def forward(self, x):
        # Extract and project features
        img_features = self.cxr_proj(self.cxr_backbone(x)).unsqueeze(1)
        clico_features = self.clico_proj(self.clico_backbone(x)).unsqueeze(1)
        clino_features = self.clino_proj(self.clino_backbone(x)).unsqueeze(1)
        # Tokenize
        tokens = self.pe(torch.cat((img_features, clico_features, clino_features), dim=1))
        # Masking, Self-Attention, and Encoding
        tokens = self.mha(query=tokens, key=tokens, value=tokens)[0] + tokens
        # Cat tokens to fused representation
        fused = F.adaptive_avg_pool1d(tokens.transpose(1, 2), 1).squeeze()
        # Classification
        out = self.classifier(fused)

        return out

    def bceloss(self,x,labels):
      loss_fct = BCEWithLogitsLoss(reduction='none')
      return(loss_fct(x.view(-1, self.num_classes), labels.float()))

    def on_train_epoch_start(self):
      self.losslist=[] ##create self.losslist at the start of each epoch then put losses to self.movingloss at the end of each epoch
      
      

    def training_step(self,batch,idx):
      
      features,y,index=batch
      scores=self.forward(features)
      loss_l =self.bceloss(scores,y)
      for ind,losses in enumerate(loss_l):  ##dicom_ids are returned in  __getitem__ method and with corresponding losses put in self.losslist
        id_loss={'dicom_id':features['dicom_id'][ind],'loss':losses.mean().cpu()}
        self.losslist.append(id_loss)
     # for pi, cl in zip(index,loss_l):      
       #  print(cl.shape)             #this is with using seed and keeping track of images with indices. 
       # self.loss_epoch[pi] = cl.mean().cpu().item()     #but lets use dicom_ids's as identifier, not indices.
      loss=loss_l.mean()    
      self.log('train_loss', loss,on_step=True, on_epoch=True)
      #self.log('train_loss', loss,on_epoch=True)
      return {"loss":loss,"scores":scores,"targets":y}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-2,weight_decay=0.01)
        self.scheduler = CustomCyclicLR(optimizer, self.r1, self.r2, self.c, self.num_cycles)
        return {"optimizer":optimizer,"lr_scheduler":{"scheduler":self.scheduler,"frequency":1,"interval": "epoch"}}

    def validation_step(self, val_batch, batch_idx):
        features,y,index = val_batch     
        scores=self.forward(features)
        loss =self.bceloss(scores,y).mean()
        self.log('val_loss', loss,on_step=True,on_epoch=True) 

    def inference(self):
      ##check
      self.entropy=[]
      for features,labels,index in self.unlabelled_loader:
        with torch.no_grad(): 
          for key,value in features.items(): ##put entire dict to gpu
            if key=='clino':
              for key,value in features['clino'].items():
                 features['clino'][key]=value.to('cuda')
            elif key!='dicom_id':
              features[key]=value.to('cuda')
          scores=self.forward(features)
          scores=torch.sigmoid(scores)
          label_entropies = -torch.sum(scores * torch.log2(scores + 1e-7), dim=1)
          for ind,entr in enumerate(label_entropies):
            id_entr={'dicom_id':features['dicom_id'][ind],'entropy':entr.cpu()}
            self.entropy.append(id_entr)
          #for pi, cl in zip(index,label_entropies):
           # self.entropy[pi]=cl.cpu()
      self.normalize_and_accumulatentropy()
      #self.moving_entropy_dic=self.moving_entropy_dic+self.entropy
          
    def on_train_end(self):
      ##on train end do CurriculumClustering to classify samples as easy,medium and hard
      ##save labelled extracted features to self.extracted_features1 
      self.extracted_features1=np.zeros((self.size_dataset,512)) 
      self.dicomlist1=[] ##save corresponding dicom_ids for images
      ##save unlabelled extracted features to self.extracted_features2
      self.extracted_features2=np.zeros((self.size_unlabelled,512))
      self.dicomlist2=[] ##save corresponding dicom_ids for images
      feature_index1 = 0
      feature_index2 = 0
      with torch.no_grad():
        for features,labels,index in self.labelled_loader:  ##feature extraction for labelled dataset
          for key,value in features.items():
            if key=='image':
              features[key]=value.to('cuda')
          outputs=self.cxr_proj(self.cxr_backbone(features))
          for ind,output in enumerate(outputs):
            self.extracted_features1[feature_index1] = output.cpu().detach().numpy()
            self.dicomlist1.append(features['dicom_id'][ind])
            feature_index1 += 1 

        for features,labels,index in self.unlabelled_loader: ##feature extraction for unlabelled dataset
          for key,value in features.items():
            if key=='image':
              features[key]=value.to('cuda')
          outputs=self.cxr_proj(self.cxr_backbone(features))
        
          for ind,output in enumerate(outputs):
            self.extracted_features2[feature_index2] = output.cpu().detach().numpy()
            self.dicomlist2.append(features['dicom_id'][ind])
            feature_index2 += 1

      self.features_concat=np.concatenate((self.extracted_features1,self.extracted_features2))
      np.save('/home/onur/features2pca.npy',self.features_concat) ##save features as npy file
      data = {'loss':self.moving_loss, 'entropy': self.moving_entropy}  ##save moving loss and entropy
      file_name='/home/onur/lossentropy2pca.pkl'
      file_name2='/home/onur/dicomlistt2pca.pkl'
      self.dicomlist=self.dicomlist1+self.dicomlist2
      with open(file_name2, 'wb') as file:
        pickle.dump(self.dicomlist, file)
      with open(file_name, 'wb') as file:
        pickle.dump(data, file)
      
      self.cluster()
      self.identifysamples()

    def cluster(self):
      n_components_list = range(100, 500, 100)
      result = []
      for n_components in n_components_list:
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(self.features_concat)
        result.append(np.sum(pca.explained_variance_ratio_))
        print(f'Cumulative explained variation for {n_components} principal components: {np.sum(pca.explained_variance_ratio_)}.')
      #tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
      #tsne_results = tsne.fit_transform(pca_result)
      n_clusters = 14
      k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
      k_means.fit(pca_result)
      Z_org = k_means.predict(pca_result)
      curriculumCluster = CurriculumClustering(verbose = True, dim_reduce = 400, calc_auxiliary =False)
      self.cu_clusters = curriculumCluster.fit_predict(self.features_concat, Z_org) ##features classified easy medium and hard
      
    def identifysamples(self):
      ## self.cu_clusters, self.moving_loss and self.moving_entropy will be used to identify noisy,informative and confident samples
      x_percentile=0.05
      y_percentile=0.05
      noisy_dicom_ids = []  # To store dicom_ids of noisy samples
      informative_dicom_ids = []  # To store dicom_ids of informative samples
      confident_dicom_ids = []  # To store dicom_ids of confident samples
      # Sort 'self.moving_loss' based on 'loss' values in descending order
      loss_ranked = sorted(range(len(self.moving_loss)), key=lambda k: -self.moving_loss[k]['loss'])
      # Select the top x_percentile percent of samples based on 'loss' for noisy samples
      x_samples = [self.moving_loss[i]['dicom_id'] for i in loss_ranked[:int(x_percentile * len(self.moving_loss))]]
      # Identify noisy samples based on 'cu_clusters' and store their 'dicom_id'
      for dicom_id in x_samples:
        index = self.dicomlist.index(dicom_id)
        if self.cu_clusters[index] == 0:
          noisy_dicom_ids.append(dicom_id)
      # Sort 'self.moving_entropy' based on 'entropy' values in ascending order
      entropy_ranked = sorted(range(len(self.moving_entropy)), key=lambda k: self.moving_entropy[k]['entropy'])
      # Select the bottom x_percentile percent of samples based on 'entropy' for confident samples
      confident_samples = [self.moving_entropy[i]['dicom_id'] for i in entropy_ranked[:int(y_percentile * len(self.moving_entropy))]]
      # Reverse the order to select the top y_percentile percent of samples for informative samples
      entropy_ranked_descending = sorted(range(len(self.moving_entropy)), key=lambda k: self.moving_entropy[k]['entropy'], reverse=True)
      informative_samples = [self.moving_entropy[i]['dicom_id'] for i in entropy_ranked_descending[:int(0.1 * len(self.moving_entropy))]]
      # Identify confident samples based on 'cu_clusters' and store their 'dicom_id'
      confident_dicom_ids = [dicom_id for dicom_id in confident_samples if self.cu_clusters[self.dicomlist.index(dicom_id)] in [0, 1]]
      # Store informative samples directly
      informative_dicom_ids = informative_samples
      print(len(informative_dicom_ids))
      print(len(confident_dicom_ids))
      print(len(noisy_dicom_ids))
      data = {'informative': informative_dicom_ids, 'confident': confident_dicom_ids, 'noisy':noisy_dicom_ids}
      file_name='/home/onur/infonoisyconfident2pca.pkl'
      with open(file_name, 'wb') as file:
        pickle.dump(data, file)
      
      
      
    def normalize_and_accumulatentropy(self): ##loss and entropy can be done in a single function
      print("entered")
      entropy_tensors = [item['entropy'] for item in self.entropy]
      total_entropy = sum(entropy.item() for entropy in entropy_tensors)
      mean_entropy = total_entropy / len(entropy_tensors)
      self.entropy = [{'dicom_id': item['dicom_id'], 'entropy': item['entropy'] - mean_entropy} for item in self.entropy]
      ##accumulate it to self.moving_entropy
      for item in self.entropy:
        dicom_id = item['dicom_id']
        normalized_entropy = item['entropy']
        found = False
        for moving_entropy_item in self.moving_entropy:
          if moving_entropy_item['dicom_id'] == dicom_id:
            moving_entropy_item['entropy'] += normalized_entropy
            found = True
            break
        if not found:
        # If the 'dicom_id' is not found, create a new dictionary
          self.moving_entropy.append({'dicom_id': dicom_id, 'entropy': normalized_entropy})
           
    #def normalize_and_accumulatentropy(self): ##loss and entropy can be done in a single function
     # entropy_tensors = [item['entropy'] for item in self.entropy]
      #total_entropy = sum(entropy.item() for entropy in entropy_tensors)
      #mean_entropy = total_entropy / len(entropy_tensors)
      #self.entropy = [{'dicom_id': item['dicom_id'], 'entropy': item['entropy'] - mean_entropy} for item in self.entropy]
      ###accumulate it to self.moving_entropy
      #for item in self.entropy:
        #dicom_id = item['dicom_id']
       # normalized_entropy = item['entropy']
        #if self.current_epoch==0:
         # self.moving_entropy.append({'dicom_id': dicom_id, 'entropy': normalized_entropy})
        #else:
         # for moving_entropy_item in self.moving_entropy:
          #  if moving_entropy_item['dicom_id'] == dicom_id:
           #   moving_entropy_item['entropy'] += normalized_entropy   
            #  break
      #if self.current_epoch ==11 or self.current_epoch == 23 or self.current_epoch ==7:
       # self.moving_entropy = sorted(self.moving_entropy, key=lambda x: x['entropy'])
        #middle_index = len(self.moving_entropy) // 2
        #num_samples_to_discard_moving = int(len(self.moving_entropy) * 0.25 / 2)
        #start_index_moving = middle_index - num_samples_to_discard_moving
        #end_index_moving = middle_index + num_samples_to_discard_moving + 1
        #self.moving_entropy = self.moving_entropy[:start_index_moving] + self.moving_entropy[end_index_moving:]
    
      
      

    def normalize_and_accumulateloss(self):
      ##normalize loss
      loss_tensors = [item['loss'] for item in self.losslist]
      total_loss = sum(loss.item() for loss in loss_tensors)
      mean_loss = total_loss / len(loss_tensors)
      self.losslist = [{'dicom_id': item['dicom_id'], 'loss': item['loss'] - mean_loss} for item in self.losslist]
      ##accumulate it to self.moving_loss
      for item in self.losslist:
        dicom_id = item['dicom_id']
        normalized_loss = item['loss']
        found = False
        for moving_loss_item in self.moving_loss:
          if moving_loss_item['dicom_id'] == dicom_id:
            moving_loss_item['loss'] += normalized_loss
            found = True
            break
        if not found:
        # If the 'dicom_id' is not found, create a new dictionary
          self.moving_loss.append({'dicom_id': dicom_id, 'loss': normalized_loss})
      
    def on_train_start(self):
        print("yes")
        for _ in range(37):
            print("xd")
            self.scheduler.step()



    def training_epoch_end(self, outputs):
      acc=[]
      f1=[]
      for output in outputs:
        scores=output["scores"]
        targets=output["targets"]
        targets = (targets >= 0.5).float()
        acc_s=accuracy(scores,targets,task='multilabel',num_labels=14)
        f1_s=f1_score(scores,targets,task='multilabel',num_labels=14)
        f1.append(f1_s)
        acc.append(acc_s)
      final_acc=sum(acc)/len(acc)
      final_f1=sum(f1)/len(f1)
      #self.loss_epoch= self.loss_epoch - self.loss_epoch.mean()

      ##normalize and accumulate loss into self.moving_loss
      self.normalize_and_accumulateloss()
      ##do inference to find predicted entropy for unlabelled samples
      current_epoch = self.trainer.current_epoch
      #if current_epoch in [45,47,48,56,57,59]:
      self.inference()
         
      self.log_dict({'accuracy':final_acc,"f1_score":final_f1})
    
    def test_epoch_end(self, outputs):
      diseases_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged_Cardiomediastinum',
                             'Fracture', 'Lung_Lesion', 'Lung_Opacity', 'No_Finding', 'Pleural_Effusion',
                             'Pleural_Other', 'Pneumonia', 'Pneumothorax', 'Support_Devices']
      for i, disease in enumerate(diseases_columns):

        all_auroc_scores = [output['auroc_scores'][i] if output['auroc_scores'] else None for output in outputs]
        all_auprc_scores = [output['auprc_scores'][i] if output['auprc_scores'] else None for output in outputs]

        all_auroc_scores = [score for score in all_auroc_scores if score is not None]
        all_auprc_scores = [score for score in all_auprc_scores if score is not None]
        
        auroc_mean = np.mean(all_auroc_scores)
        auprc_mean = np.mean(all_auprc_scores)
        print(f'{disease}: AUROC = {auroc_mean:.4f}, AUPRC = {auprc_mean:.4f}')


    def test_step(self, batch, batch_idx):
        # Your test step logic here
      features,targets,index = batch
      x=features['image']
      text=self.clino_backbone(features)
      lab=torch.unsqueeze(features['clico'],2)
      sex=features['demog'][:,0]
      sex=sex.view(-1,1,1)
      age=features['demog'][:,1]
      age=age.view(-1,1,1) 
      scores=self.forward(x,text,lab,sex,age)
      outputs=torch.sigmoid(scores)

      outputs_np = outputs.detach().cpu().numpy()
      targets_np = targets.detach().cpu().numpy()

        # Diseases order
      diseases_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged_Cardiomediastinum',
                             'Fracture', 'Lung_Lesion', 'Lung_Opacity', 'No_Finding', 'Pleural_Effusion',
                             'Pleural_Other', 'Pneumonia', 'Pneumothorax', 'Support_Devices']

      auroc_scores = []
      auprc_scores = []

        # Calculate AUROC and AUPRC for each disease
      for i, disease in enumerate(diseases_columns):
        try:
          auroc = roc_auc_score(targets_np[:, i], outputs_np[:, i]) 
          auroc_scores.append(auroc)
          precisions, recalls, _ = precision_recall_curve(targets_np[:, i], outputs_np[:, i])
          auprc = auc(recalls, precisions)
          auprc_scores.append(auprc)
        except ValueError as e:
          print(f"Skipping {disease} due to: {e}")
          auroc_scores.append(None)
          auprc_scores.append(None)
      return {'auroc_scores': auroc_scores, 'auprc_scores': auprc_scores}
      


        
