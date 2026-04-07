"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
import torchmetrics
import wandb
logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 32
    learning_rate = 3e-5
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    
    def __init__(self, model, train_dataset, test_dataset, config, writer=None):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.summaryWriter = writer

        # take over whatever gpus are on the system
        self.device = 'gpu'
        
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=False, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)
            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, batch in pbar:
                # Support both (x, y, m) and (x, y, m, e) tuples
                if len(batch) == 4:
                    x, y, m, e = batch
                    e = e.to(self.device)
                else:
                    x, y, m = batch
                    e = None
                #print('x:', x.shape, 'y:', y.shape, 'm:',  m.shape)
    
                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                m = m.to(self.device)
                
                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, _ = model(x, y, m, eigen=e)
                    mask = y != 25  # Create a boolean mask for non-pad tokens
                    logits_masked = logits[mask]
                    y_masked = y[mask]
                    
                    loss = torch.nn.functional.cross_entropy(logits_masked, y_masked)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())
                
                    # Calculate accuracy
                    predictions = logits.permute(0, 2, 1)
                
                    # Create an indexing tensor based on the mask
                    mask_indices = torch.where(mask)

                    # Use the indexing tensor to filter predictions and ground_truth
                    predictions = predictions[mask_indices[0], :, mask_indices[1]]
                    ground_truth = y[mask_indices]
                    
                    acc_top1 = torchmetrics.functional.accuracy(
                        predictions, 
                        ground_truth, 
                        task="multiclass", 
                        num_classes=self.train_dataset.vocab_size, 
                        top_k=1
                    )
                    acc_top3 = torchmetrics.functional.accuracy(
                        predictions, 
                        ground_truth, 
                        task="multiclass", 
                        num_classes=self.train_dataset.vocab_size, 
                        top_k=3
                    )
                    acc_top5 = torchmetrics.functional.accuracy(
                        predictions, 
                        ground_truth, 
                        task="multiclass", 
                        num_classes=self.train_dataset.vocab_size, 
                        top_k=5
                    )

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    self.summaryWriter.add_scalar("batchLoss", loss.item(),(len(loader)*epoch)+it)
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if is_train:
                train_loss = float(np.mean(losses))
                if self.summaryWriter:
                    self.summaryWriter.add_scalar("epochLoss", train_loss, epoch)
                logger.info("epoch train loss: %f", train_loss)
                # log metrics to wandb
                print("train loss:", train_loss)
                wandb.log({"Train loss": train_loss})
                wandb.log({"hits@1": acc_top1})
                wandb.log({"hits@3": acc_top3})
                wandb.log({"hits@5": acc_top5})
                return train_loss
            
            if not is_train:
                test_loss = float(np.mean(losses))
                if self.summaryWriter:
                    self.summaryWriter.add_scalar("valLoss", test_loss, epoch)
                logger.info("test loss: %f", test_loss)
                wandb.log({"Validation loss": test_loss})
                return test_loss
      

        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch('test')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()

