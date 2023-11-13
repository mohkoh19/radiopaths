import os
import traceback
from argparse import ArgumentParser
from datasets.mimic_cxr import MIMIC_CXR_Loader
from pytorch_lightning import Trainer
from models.clino import ClinoClassifier
from models.modeling_irene import IRENE,CONFIGS
from models2.multimodal import Radiopaths
from torchvision.utils import save_image
import models.configs as configs
from torch.nn import BCEWithLogitsLoss
from pytorch_lightning.loggers import WandbLogger
import torch
from pytorch_lightning import Trainer
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
import pandas as pd
if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--model_name",type=str,default="IRENE",help="Clino,IRENE,Radiopaths")
    parser.add_argument("--traintest",type=str,default="train",help="train,test")
    parser.add_argument("--exp_name", type=str, default=None, help="Name of run")
    parser.add_argument( "--ckpt_path",type=str,default=None,help="Path to checkpoint of model to resume",)
    parser.add_argument("--save_dir",  type=str, default="./",help="Directory to save logs and checkpoints",)
    parser = MIMIC_CXR_Loader.add_argparse_args(parser)
    args = parser.parse_args()

    ckpt_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    dm = MIMIC_CXR_Loader.from_argparse_args(args)
    dm.prepare_data()
    dm.setup()
    trainn=dm._get_dataloader("train",1)
    unlabelled_loader=dm._get_dataloader("train",0)
    vall=dm._get_dataloader("val",1)
    testt=dm._get_dataloader("test",1)
    unlabelled_dataset=dm.return_dataset()
    print("lengthtraindataset",len(trainn.dataset))
    print("lengthunlabelled",len(unlabelled_dataset))
    
    

    if args.model_name=="IRENE":
      config = CONFIGS["IRENE"]
      model = IRENE(config,len(trainn.dataset), 224,zero_head=True, num_classes=14)
    elif args.model_name=="Clino":
      model = ClinoClassifier()
    elif args.model_name=="Radiopaths":
      model=Radiopaths(len(trainn.dataset),len(unlabelled_dataset),unlabelled_loader,trainn)

    

    checkpoint_callback = ModelCheckpoint(monitor="val_loss",  save_top_k=4,  mode='min',  dirpath=ckpt_dir, filename=args.exp_name
        + "-epoch={epoch}-"
        + 'val_loss'.replace("/", "_")
        + "={"
        + 'val_loss'
        + ":.2f}",
        auto_insert_metric_name=False,)
    checkpoint_callback2 = ModelCheckpoint(mode='min',  dirpath=ckpt_dir, filename=args.exp_name
        + "-epoch={epoch}-"
        + 'val_loss'.replace("/", "_")
        + "={"
        + 'val_loss'
        + ":.2f}",
        auto_insert_metric_name=False,every_n_epochs=29)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    if args.traintest=="train":

      wandb_logger = WandbLogger(project="pathal",name=args.exp_name,log_model="all",save_dir=args.save_dir)
      trainer = Trainer(logger=wandb_logger,max_epochs=30,callbacks=[checkpoint_callback,checkpoint_callback2,lr_monitor],accelerator='gpu')
      wandb_logger.watch(model, log="all", log_graph=False)
      try:
          trainer.fit(model,trainn,vall,ckpt_path=None if args.ckpt_path is None else args.ckpt_path)
          wandb_logger.finalize("success")
          wandb.finish(0)
      except KeyboardInterrupt as ki:
          print(ki)
          wandb_logger.finalize("abort")
          wandb.finish(-1)
      except Exception:
          traceback.print_exc()
          wandb_logger.finalize("error")
          wandb.finish(-1)
    
    elif args.traintest=="test": 
      trainer=Trainer(accelerator="gpu")
      trainer.test(model,testt,ckpt_path=args.ckpt_path)

    
    
     