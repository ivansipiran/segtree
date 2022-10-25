import argparse
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import os
import glob 
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

T = transforms.ToPILImage()

#Building dataloaders
class TreeDataset(torch.utils.data.Dataset):
  def __init__(self, root, split, transform=None):
    self.root = root
    self.split = split
    self.transform = transform

    self.folder_images = os.path.join(self.root, split+'_imgs', split)
    self.folder_masks = os.path.join(self.root, split+'_masks', split)

    self.list_images = sorted(glob.glob(self.folder_images + '/*.jpg'))
    self.list_masks = sorted(glob.glob(self.folder_masks + '/*.png'))
        
  def __len__(self):
    return len(self.list_images)
  
  def __getitem__(self, idx):
    stem = self.list_masks[idx].split('/')[5]

    img = np.array(Image.open(self.list_images[idx]).convert('RGB'))
    mask = np.array(Image.open(self.list_masks[idx]))
    mask = mask.astype(np.float32)
    mask = mask/255.

    if self.transform:
      img = self.transform(img)
      mask = self.transform(mask)

    size = np.array(mask).shape
    img = np.array(Image.fromarray(img).resize((256, 256), Image.LINEAR))
    mask = np.array(Image.fromarray(mask).resize((256, 256), Image.NEAREST))
    
    img = np.moveaxis(img, -1, 0)
    mask = np.expand_dims(mask, 0)

    sample = dict(image=img, mask=mask, stem=stem, original_size=size)
    return sample

class SegTreeModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.epoch=0

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        
        self.log_dict(metrics, prog_bar=True)
        if stage=='valid':
            print(f'Epoch:{self.epoch}-IOU{dataset_iou}')
            self.epoch=self.epoch+1

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


def test_file(checkpoint, output_dir, save_results=False):
    #checkpoint='chekpoint-unet-efficientnet-b3-epoch=40-valid_dataset_iou=0.6281.ckpt'
    name_checkpoint = checkpoint.split('/')[-1]
    folder = "-".join(name_checkpoint.split('-')[1:])
    folder = "".join(folder.split(".")[:-1])

    part1 = name_checkpoint.split('-epoch')[0]
    L = part1.split('-')
    network = L[1]
    
    if len(L)==3:
        encoder = L[2]
    if len(L)==4:
        encoder = '-'.join([L[2], L[3]])

    if not os.path.isdir(os.path.join(output_dir,folder)):
        os.mkdir(os.path.join(output_dir,folder))


    test_dataset = TreeDataset(root=os.path.join('/', 'data', 'Trees'), split='test')
    n_cpu = 0
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=n_cpu)

    model = SegTreeModel(network, encoder, in_channels=3, out_classes=1)
    checkp = torch.load(checkpoint)
    model.load_state_dict(checkp["state_dict"])

    batch = next(iter(test_dataloader))
    with torch.no_grad():
        model.eval()
        logits = model(batch["image"])
    pr_masks = logits.sigmoid()

    model.eval()
    for batch in test_dataloader:
        with torch.no_grad():
            logits = model(batch["image"])
        pr_masks = logits.sigmoid()
        pr_masks = (255*(pr_masks > 0.5)).type(torch.uint8)

        for image, gt_mask, pr_mask, name, size in zip(batch["image"], batch["mask"], pr_masks, batch["stem"], batch["original_size"]):
            back_im = T(pr_mask).resize(size, Image.NEAREST)
            back_im.save(os.path.join(output_dir, folder, name))

            if save_results:
                plt.figure(figsize=(10, 5))

                plt.subplot(1, 3, 1)
                plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
                plt.title("Image")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(gt_mask.numpy().squeeze()) # just squeeze classes dim, because we have only one class
                plt.title("Ground truth")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.imshow(pr_mask.numpy().squeeze()) # just squeeze classes dim, because we have only one class
                plt.title("Prediction")
                plt.axis("off")

                plt.savefig(os.path.join(output_dir, folder, 'res_'+name))

def test_dir(checkpoint_dir, output_dir, save_results=False):
    L = os.listdir(checkpoint_dir)

    for l in L:
        test_file(os.path.join(checkpoint_dir,l), output_dir, save_results)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint_dir', type=str)
    group.add_argument('--checkpoint_file', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--save_results', action='store_true')

    opt = parser.parse_args()
    
    if opt.checkpoint_dir is not None:
        test_dir(opt.checkpoint_dir, opt.output_dir, opt.save_results)
    
    if opt.checkpoint_file is not None:
        test_file(opt.checkpoint_file, opt.output_dir, opt.save_results)
    
       