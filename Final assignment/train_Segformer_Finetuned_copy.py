"""
Training script for SegFormer with FLOPs, mean Dice, and mixed precision.
"""
import os
from argparse import ArgumentParser
import wandb
torch_import = "torch"
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToImage, ToDtype, RandomHorizontalFlip, RandomRotation, ColorJitter, GaussianBlur
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig
from ptflops import get_model_complexity_info
from torch.cuda.amp import autocast, GradScaler

# ID mappings
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0,0,0)

def convert_to_train_id(x): return x.apply_(lambda v: id_to_trainid[v])

def compute_flops(model, res): macs,_=get_model_complexity_info(model,res,as_strings=False,print_per_layer_stat=False,verbose=False); return macs*2

def dice_per_class(pred, tgt, n):
    pred, tgt = pred.view(-1), tgt.view(-1)
    d = torch.zeros(n,device=pred.device,dtype=torch.float64)
    for c in range(n):
        if c==255: continue
        p, t = pred==c, tgt==c; denom=(p.sum()+t.sum())
        d[c] = 2*(p&t).sum()/denom if denom>0 else 0
    return d

# Args parser

def get_args_parser():
    p=ArgumentParser("SegFormer mixed precision");
    p.add_argument("--data-dir",default="./data/cityscapes"); p.add_argument("--batch-size",type=int,default=64);
    p.add_argument("--epochs",type=int,default=10); p.add_argument("--lr",type=float,default=1e-3);
    p.add_argument("--num-workers",type=int,default=10); p.add_argument("--seed",type=int,default=42);
    p.add_argument("--experiment-id",default="segformer-mp"); return p

# Main

def main(args):
    wandb.init(project="5lsm0-cityscapes-segmentation-loss-combination",name=args.experiment_id,config=vars(args))
    torch.manual_seed(args.seed); torch.backends.cudnn.deterministic=True
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler=GradScaler()

    # Data transforms
    proc=SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    tr=Compose([ToImage(),RandomHorizontalFlip(0.5),Resize((1024,1024)),ColorJitter(0.3,0.3,0.3,0.1),ToDtype(torch.float32,True),RandomRotation(30),GaussianBlur(3,(0.1,2.0)),Normalize(proc.image_mean,proc.image_std)])
    va=Compose([ToImage(),Resize((1024,1024)),ToDtype(torch.float32,True),Normalize(proc.image_mean,proc.image_std)])
    train_ds=wrap_dataset_for_transforms_v2(Cityscapes(args.data_dir,"train","fine","semantic",tr))
    valid_ds=wrap_dataset_for_transforms_v2(Cityscapes(args.data_dir,"val","fine","semantic",va))
    tl=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
    vl=DataLoader(valid_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)

    # Model
    cfg=SegformerConfig.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024",num_channels=3,num_labels=19,upsample_ratio=1)
    model=SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024",config=cfg,ignore_mismatched_sizes=True).to(device)
    criterion=smp.losses.DiceLoss(mode="multiclass",ignore_index=255)
    dice_fn=smp.losses.DiceLoss(mode="multiclass",ignore_index=255)
    # Optimizer
    enc,dec=[],[]
    for n,p in model.named_parameters(): (enc if "encoder" in n else dec).append(p)
    opt=AdamW([{"params":enc,"lr":args.lr*0.1},{"params":dec,"lr":args.lr}])
    sch=CosineAnnealingLR(opt,T_max=args.epochs,eta_min=1e-7)

    best_loss=float('inf')
    os.makedirs(os.path.join("checkpoints",args.experiment_id),exist_ok=True)

    # Train/Val
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        model.train()
        for i,(imgs,lbls) in enumerate(tl):
            lbls=convert_to_train_id(lbls).to(device).long().squeeze(1)
            imgs=imgs.to(device)
            opt.zero_grad()
            with autocast():
                logits=model(imgs).logits
                logits=torch.nn.functional.interpolate(logits,size=lbls.shape[1:],mode="bilinear",align_corners=False)
                loss=criterion(logits,lbls)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            wandb.log({"train_loss":loss.item(),"lr":opt.param_groups[1]["lr"],"epoch":epoch+1},step=epoch*len(tl)+i)
        # Validation
        model.eval(); vloss,vdice=[],[]
        with torch.no_grad():
            for imgs,lbls in vl:
                lbls=convert_to_train_id(lbls).to(device).long().squeeze(1)
                imgs=imgs.to(device)
                with autocast():
                    logits=model(imgs).logits
                    logits=torch.nn.functional.interpolate(logits,size=lbls.shape[1:],mode="bilinear",align_corners=False)
                l=criterion(logits,lbls); vloss.append(l.item()); vdice.append(dice_fn(logits,lbls).item())
        val_loss=sum(vloss)/len(vloss); val_dice=sum(vdice)/len(vdice)
        sch.step()
        wandb.log({"valid_loss":val_loss,"valid_dice_loss":val_dice},step=(epoch+1)*len(tl)-1)
        if val_loss<best_loss: best_loss=val_loss; torch.save(model.state_dict(),os.path.join("checkpoints",args.experiment_id,"best.pth"))

    print("Training complete")
    # Final metrics
    model.eval(); ncls=19; dsum=torch.zeros(ncls,dtype=torch.float64)
    cnt=0
    with torch.no_grad():
        for imgs,lbls in vl:
            lbls=convert_to_train_id(lbls).to(device).long().squeeze(1)
            imgs=imgs.to(device)
            with autocast():
                preds=model(imgs).logits.argmax(1)
            dsum+=dice_per_class(preds,lbls,ncls).cpu(); cnt+=1
    mean_dice=dsum/cnt; flops=compute_flops(model.to("cpu"),(3,1024,1024))/1e9
    table=wandb.Table(columns=["Metric","Value"]); table.add_data("FLOPs(G)",f"{flops:.2f}")
    for c in range(ncls): table.add_data(f"Dice_{c}",f"{mean_dice[c]:.4f}")
    wandb.log({"FLOPs_and_MeanDice":table}); wandb.finish()

if __name__=="__main__": main(get_args_parser().parse_args())
