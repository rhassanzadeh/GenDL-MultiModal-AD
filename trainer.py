import json, argparse, time, os, shutil, itertools
import pandas as pd
import numpy as np
from os.path import join, exists
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import *
from data_loader import sMRIDataset, FNCDataset
from models.networks import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")



class Trainer(object):
    def __init__(self, trial_dir: str, config: argparse.Namespace):
        self.trial_dir = trial_dir
        self.config = config
        self.writer = SummaryWriter(join(self.trial_dir, f'tensorboard_r{self.config.repeat_num}s{self.config.split_num}'))

        # Set device on GPU if available, else CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
    def setup_model(self):
        """ 
        Set up model and loss function.
        """
        
        G_sMRI2FNC = GeneratorsMRItoFNC().to(self.device)
        G_FNC2sMRI = GeneratorFNCtosMRI().to(self.device)
        D_sMRI = DiscriminatorsMRI().to(self.device)
        D_FNC = DiscriminatorFNC().to(self.device)
        
        criterion_GAN = nn.MSELoss().to(self.device)
        criterion_cycle = nn.L1Loss().to(self.device)
          
        return (G_sMRI2FNC, G_FNC2sMRI, D_sMRI, D_FNC, criterion_GAN, criterion_cycle)

    
    def setup_optimizer(self, G_sMRI2FNC, G_FNC2sMRI, D_sMRI, D_FNC):
        """ 
        Set up optimizer for network
        """
        
        optimizer_G = optim.Adam(
            itertools.chain(G_sMRI2FNC.parameters(), G_FNC2sMRI.parameters()), 
            lr=self.config.learning_rate, betas=(self.config.beta1, 0.999), weight_decay = self.config.weight_decay,
        )
        optimizer_D = optim.Adam(
            itertools.chain(D_sMRI.parameters(), D_FNC.parameters()), 
            lr=self.config.learning_rate, betas=(self.config.beta1, 0.999), weight_decay= self.config.weight_decay,
        )
        
        return (optimizer_G, optimizer_D)
    

    def train(self):  
        X_train_valid = pd.read_csv(f'{self.config.data_path}/train_r{self.config.repeat_num}s{self.config.split_num}.csv')
        unique_subj = X_train_valid.drop_duplicates(subset="SubID").reset_index(drop=True)
        train_subs, valid_subs = train_test_split(
            unique_subj['SubID'], test_size=0.2, stratify=unique_subj['Diagnosis'], random_state=self.config.random_seed,
        )
        X_train = X_train_valid[X_train_valid['SubID'].isin(train_subs)]
        X_valid = X_train_valid[X_train_valid['SubID'].isin(valid_subs)]
        
        # Build data loaders
        sMRI_train_loader = DataLoader(
            dataset=sMRIDataset(data=X_train.dropna(subset=['sMRI_path'])), 
            batch_size=self.config.batch_size, num_workers=self.config.num_workers, 
            shuffle=True, pin_memory=True, drop_last=(len(X_train)%self.config.batch_size == 1), 
        )
        sMRI_valid_loader = DataLoader(
            dataset=sMRIDataset(data=X_valid.dropna(subset=['sMRI_path'])), 
            batch_size=self.config.batch_size, num_workers=self.config.num_workers, 
            shuffle=False, pin_memory=True, drop_last=False, 
        )
        FNC_train_loader = DataLoader(
            dataset=FNCDataset(data=X_train.dropna(subset=['FNC_path'])), 
            batch_size=self.config.batch_size, num_workers=self.config.num_workers, 
            shuffle=True, pin_memory=True, drop_last=(len(X_train)%self.config.batch_size == 1), 
        )
        FNC_valid_loader = DataLoader(
            dataset=FNCDataset(data=X_valid.dropna(subset=['FNC_path'])), 
            batch_size=self.config.batch_size, num_workers=self.config.num_workers, 
            shuffle=False, pin_memory=True, drop_last=False, 
        )
        FNC_train_loader = itertools.cycle(FNC_train_loader)
        FNC_valid_loader = itertools.cycle(FNC_valid_loader)
        num_train = len(X_train.dropna(subset=['sMRI_path']))

        # Early stopping params
        self.best_valid_perf = None
        self.counter = 0

        # Build networks and criterions
        self.G_sMRI2FNC, self.G_FNC2sMRI, self.D_sMRI, self.D_FNC, self.criterion_GAN, self.criterion_cycle = self.setup_model()
        # Build optimizers
        self.optimizer_G, self.optimizer_D = self.setup_optimizer(self.G_sMRI2FNC, self.G_FNC2sMRI, self.D_sMRI, self.D_FNC)

        self.start_epoch = 1 
        for epoch in range(self.start_epoch, self.config.epochs + 1):

            train_loss, train_time = self.train_one_epoch(epoch, sMRI_train_loader, FNC_train_loader, num_train)
            valid_loss = self.validate(epoch, sMRI_valid_loader, FNC_valid_loader)
                                                
            # Check for improvement
            if self.best_valid_perf is None:
                self.best_valid_perf = valid_loss
                is_best = True
            else:
                is_best = valid_loss < self.best_valid_perf          
            msg = "Epoch:{}, {:.1f}s - train loss: {:.6f} - validation loss: {:.6f}"
            if is_best:
                msg += " [*]"
                self.counter = 0
            print(msg.format(epoch, train_time, train_loss, valid_loss))

            # Checkpoint the model
            if not is_best:
                self.counter += 1
            if self.counter > self.config.train_patience and self.config.early_stop:
                print("[!] No improvement in a while, stopping training.")
                break
            self.best_valid_perf = min(valid_loss, self.best_valid_perf)
            self.save_checkpoint(
            {
                'epoch': epoch,
                'G_sMRI2FNC_state': self.G_sMRI2FNC.state_dict(),
                'G_FNC2sMRI_state': self.G_FNC2sMRI.state_dict(),
                'D_sMRI_state': self.D_sMRI.state_dict(),
                'D_FNC_state': self.D_FNC.state_dict(),
                'optimizer_G_state': self.optimizer_G.state_dict(), 
                'optimizer_D_state': self.optimizer_D.state_dict(), 
                'best_valid_perf': self.best_valid_perf,
                'last_valid_perf': valid_loss,
                'counter': self.counter,
            }, repeat_num=self.config.repeat_num, split_num=self.config.split_num, is_best=is_best)
                
            self.writer.flush()
            self.writer.close()

        print("\ndone!")
        

    def train_one_epoch(self, epoch, sMRI_train_loader, FNC_train_loader, num_train):
        batch_time = AverageMeter()
        batch_time.reset()
        losses = AverageMeter()
        losses.reset()
        tic = time.time()
        
        # switch to train mode
        self.G_sMRI2FNC.train()
        self.G_FNC2sMRI.train()
        self.D_sMRI.train()
        self.D_FNC.train()
        
        with tqdm(total=num_train) as pbar:  
            for batch_index, (real_sMRI, real_FNC) in enumerate(zip(sMRI_train_loader, FNC_train_loader)): 
                batch_size = real_sMRI.shape[0]
                
                real_sMRI = real_sMRI.to(self.device)
                real_FNC = real_FNC.to(self.device)
                
                # Train Generators
                self.optimizer_G.zero_grad()
                
                fake_FNC = self.G_sMRI2FNC(real_sMRI)
                fake_sMRI = self.G_FNC2sMRI(real_FNC)
                
                if batch_index % 10 == 0:
                    slice_no = 60 # the 60th slice
                    
                    real_sMRI_mean = torch.mean(real_sMRI, dim=0).detach().cpu().numpy()
                    fake_sMRI_mean = torch.mean(fake_sMRI, dim=0).detach().cpu().numpy()
                    
                    self.writer.add_image('Train/sMRI_Real', real_sMRI_mean[:,:,slice_no,:], global_step=batch_index + (epoch-1))
                    self.writer.add_image('Train/sMRI_Fake', fake_sMRI_mean[:,:,slice_no,:], global_step=batch_index + (epoch-1))
                    
                    real_FNC_mean = torch.mean(real_FNC, dim=0).detach().cpu().numpy()
                    real_FNC_heatmap = np.zeros((53, 53))
                    real_FNC_heatmap[np.triu_indices(53, 1)] = real_FNC_mean
                    real_FNC_heatmap = real_FNC_heatmap + real_FNC_heatmap.T
                    fake_FNC_heatmap = np.zeros((53, 53))
                    fake_FNC_mean = torch.mean(fake_FNC, dim=0).detach().cpu().numpy()
                    fake_FNC_heatmap[np.triu_indices(53, 1)] = fake_FNC_mean
                    fake_FNC_heatmap = fake_FNC_heatmap + fake_FNC_heatmap.T

                    real_FNC_heatmap_tensor = apply_colormap(real_FNC_heatmap)
                    fake_FNC_heatmap_tensor = apply_colormap(fake_FNC_heatmap)

                    self.writer.add_image('Train/FNC_Real', real_FNC_heatmap_tensor, global_step=batch_index + (epoch-1))
                    self.writer.add_image('Train/FNC_Fake', fake_FNC_heatmap_tensor, global_step=batch_index + (epoch-1))
                    
                
                # GAN loss
                loss_GAN_sMRI2FNC = self.criterion_GAN(self.D_FNC(fake_FNC), torch.ones_like(self.D_FNC(fake_FNC)))
                loss_GAN_FNC2sMRI = self.criterion_GAN(self.D_sMRI(fake_sMRI), torch.ones_like(self.D_sMRI(fake_sMRI)))

                # Cycle loss
                recovered_sMRI = self.G_FNC2sMRI(fake_FNC)
                recovered_FNC = self.G_sMRI2FNC(fake_sMRI)
                loss_cycle_sMRI = self.criterion_cycle(recovered_sMRI, real_sMRI)
                loss_cycle_FNC = self.criterion_cycle(recovered_FNC, real_FNC)
                
                # Total loss
                loss_G = (loss_GAN_sMRI2FNC + loss_GAN_FNC2sMRI + loss_cycle_sMRI + loss_cycle_FNC) * 0.25
                loss_G.backward()
                self.optimizer_G.step()
                
                # Train Discriminators
                self.optimizer_D.zero_grad()
                
                loss_D_sMRI_real = self.criterion_GAN(self.D_sMRI(real_sMRI), torch.ones_like(self.D_sMRI(real_sMRI)))
                loss_D_sMRI_fake = self.criterion_GAN(self.D_sMRI(fake_sMRI.detach()), torch.zeros_like(self.D_sMRI(fake_sMRI)))
                loss_D_sMRI = (loss_D_sMRI_real + loss_D_sMRI_fake) * 0.5
                loss_D_sMRI.backward()
                
                loss_D_FNC_real = self.criterion_GAN(self.D_FNC(real_FNC), torch.ones_like(self.D_FNC(real_FNC)))
                loss_D_FNC_fake = self.criterion_GAN(self.D_FNC(fake_FNC.detach()), torch.zeros_like(self.D_FNC(fake_FNC)))
                loss_D_FNC = (loss_D_FNC_real + loss_D_FNC_fake) * 0.5
                loss_D_FNC.backward()
                
                loss_D = (loss_D_sMRI + loss_D_FNC) * 0.5  
                
                self.optimizer_D.step()
                    
                    
                loss = (loss_G + loss_D) * 0.5
                
                # update metric
                losses.update(loss.item(), batch_size)
                
                # measure elapsed time
                toc = time.time()
                batch_time.update(toc-tic)
                tic = time.time()
                
                pbar.set_description(("{:.1f}s - loss: {:.3f}".format(batch_time.val, losses.val)))
                pbar.update(batch_size)
                
                self.writer.add_scalar('Train/loss_GAN_sMRI2FNC', loss_GAN_sMRI2FNC.item(), global_step=batch_index + (epoch-1))
                self.writer.add_scalar('Train/loss_GAN_FNC2sMRI', loss_GAN_FNC2sMRI.item(), global_step=batch_index + (epoch-1))
                self.writer.add_scalar('Train/loss_cycle_sMRI', loss_cycle_sMRI.item(), global_step=batch_index + (epoch-1))
                self.writer.add_scalar('Train/loss_cycle_FNC', loss_cycle_FNC.item(), global_step=batch_index + (epoch-1))
                self.writer.add_scalar('Train/loss_G', loss_D.item(), global_step=batch_index + (epoch-1))
                self.writer.add_scalar('Train/loss_D_sMRI_real', loss_D_sMRI_real.item(), global_step=batch_index + (epoch-1))
                self.writer.add_scalar('Train/loss_D_sMRI_fake', loss_D_sMRI_fake.item(), global_step=batch_index + (epoch-1))
                self.writer.add_scalar('Train/loss_D_FNC_real', loss_D_FNC_real.item(), global_step=batch_index + (epoch-1))
                self.writer.add_scalar('Train/loss_D_FNC_fake', loss_D_FNC_fake.item(), global_step=batch_index + (epoch-1))
                self.writer.add_scalar('Train/loss_D', loss_D.item(), global_step=batch_index + (epoch-1))
                self.writer.add_scalar('Train/loss_total', loss.item(), global_step=batch_index + (epoch-1))
             
        # Write in tensorboard
        self.writer.add_scalar('Loss/train', losses.avg, epoch)

        return losses.avg, batch_time.sum
    

    def validate(self, epoch, sMRI_valid_loader, FNC_valid_loader):
        losses = AverageMeter()
        losses.reset()
        
        # switch to evaluate mode
        self.G_sMRI2FNC.eval()
        self.G_FNC2sMRI.eval()
        self.D_sMRI.eval()
        self.D_FNC.eval()
        
        with torch.no_grad():
            for batch_index, (real_sMRI, real_FNC) in enumerate(zip(sMRI_valid_loader, FNC_valid_loader)): 
                batch_size = real_sMRI.shape[0]
                
                real_sMRI = real_sMRI.to(self.device)
                real_FNC = real_FNC.to(self.device)
                
                fake_FNC = self.G_sMRI2FNC(real_sMRI)
                fake_sMRI = self.G_FNC2sMRI(real_FNC)
                
                if batch_index % 10 == 0:
                    slice_no = 60 # the 60th slice
                    
                    real_sMRI_mean = torch.mean(real_sMRI, dim=0).detach().cpu().numpy()
                    fake_sMRI_mean = torch.mean(fake_sMRI, dim=0).detach().cpu().numpy()
                    
                    self.writer.add_image('Valid/sMRI_Real', real_sMRI_mean[:,:,slice_no,:], global_step=batch_index + (epoch-1))
                    self.writer.add_image('Valid/sMRI_Fake', fake_sMRI_mean[:,:,slice_no,:], global_step=batch_index + (epoch-1))
                    
                    real_FNC_mean = torch.mean(real_FNC, dim=0).detach().cpu().numpy()
                    real_FNC_heatmap = np.zeros((53, 53))
                    real_FNC_heatmap[np.triu_indices(53, 1)] = real_FNC_mean
                    real_FNC_heatmap = real_FNC_heatmap + real_FNC_heatmap.T
                    fake_FNC_heatmap = np.zeros((53, 53))
                    fake_FNC_mean = torch.mean(fake_FNC, dim=0).detach().cpu().numpy()
                    fake_FNC_heatmap[np.triu_indices(53, 1)] = fake_FNC_mean
                    fake_FNC_heatmap = fake_FNC_heatmap + fake_FNC_heatmap.T

                    real_FNC_heatmap_tensor = apply_colormap(real_FNC_heatmap)
                    fake_FNC_heatmap_tensor = apply_colormap(fake_FNC_heatmap)

                    self.writer.add_image('Valid/FNC_Real', real_FNC_heatmap_tensor, global_step=batch_index + (epoch-1))
                    self.writer.add_image('Valid/FNC_Fake', fake_FNC_heatmap_tensor, global_step=batch_index + (epoch-1))

                
                # GAN loss
                loss_GAN_sMRI2FNC = self.criterion_GAN(self.D_FNC(fake_FNC), torch.ones_like(self.D_FNC(fake_FNC)))
                loss_GAN_FNC2sMRI = self.criterion_GAN(self.D_sMRI(fake_sMRI), torch.ones_like(self.D_sMRI(fake_sMRI)))

                # Cycle loss
                recovered_sMRI = self.G_FNC2sMRI(fake_FNC)
                recovered_FNC = self.G_sMRI2FNC(fake_sMRI)
                loss_cycle_sMRI = self.criterion_cycle(recovered_sMRI, real_sMRI)
                loss_cycle_FNC = self.criterion_cycle(recovered_FNC, real_FNC)
                
                loss_G = (loss_GAN_sMRI2FNC + loss_GAN_FNC2sMRI + loss_cycle_sMRI + loss_cycle_FNC) * 0.25
                
                # Discriminator sMRI loss
                loss_D_sMRI_real = self.criterion_GAN(self.D_sMRI(real_sMRI), torch.ones_like(self.D_sMRI(real_sMRI)))
                loss_D_sMRI_fake = self.criterion_GAN(self.D_sMRI(fake_sMRI.detach()), torch.zeros_like(self.D_sMRI(fake_sMRI)))
                loss_D_sMRI = (loss_D_sMRI_real + loss_D_sMRI_fake) * 0.5
                
                # Discriminator FNC loss
                loss_D_FNC_real = self.criterion_GAN(self.D_FNC(real_FNC), torch.ones_like(self.D_FNC(real_FNC)))
                loss_D_FNC_fake = self.criterion_GAN(self.D_FNC(fake_FNC.detach()), torch.zeros_like(self.D_FNC(fake_FNC)))
                loss_D_FNC = (loss_D_FNC_real + loss_D_FNC_fake) * 0.5
                
                loss_D = (loss_D_sMRI + loss_D_FNC) * 0.5
                
                loss = (loss_G + loss_D) * 0.5
                
                # update metric
                losses.update(loss.item(), batch_size)
                

            # Write in tensorboard
            self.writer.add_scalar('Valid/loss', losses.avg, epoch)

        return losses.avg


    def test(self): 
        pass
            
    
    def save_checkpoint(self, state, repeat_num, split_num, is_best):            
        filename = f'model_ckpt_r{repeat_num}s{split_num}.tar'
        ckpt_path = join(self.trial_dir, filename)
        torch.save(state, ckpt_path)
        
        if is_best:
            filename = f'best_model_ckpt_r{repeat_num}s{split_num}.tar'
            shutil.copyfile(ckpt_path, join(self.trial_dir, filename))
            
              