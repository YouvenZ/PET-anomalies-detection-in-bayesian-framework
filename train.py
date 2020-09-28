import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from load_LIDC_data import LIDC_IDRI
from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = LIDC_IDRI(dataset_location = 'data/')
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(dataset, batch_size=5, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)
print("Number of training/test patches:", (len(train_indices),len(test_indices)))

net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=2, no_convs_fcomb=4, beta=10.0)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
epochs = 10
for epoch in range(epochs):
    for step, (patch, mask, _) in enumerate(train_loader): 
        patch = patch.to(device)
        mask = mask.to(device)
        mask = torch.unsqueeze(mask,1)
        net.forward(patch, mask, training=True)
        elbo = net.elbo(mask)
        reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
        loss = -elbo + 1e-5 * reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


checkpoint_path = './chkpoint_'
best_model_path = './bestmodel.pt'
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
net.to(device)
net.train()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
epochs = 10

valid_loss_min=float('inf')
train_loss,val_loss=[],[]
dice_score_train,dice_score_val=[],[]


for epoch in range(epochs):
    running_train_loss = []
    running_train_score = []
    print('Numbers of epoch:{}/{}'.format(epoch+1,epochs))
    starded = time.time()
          
    for batch_idx, (data, target) in enumerate(train_loader):
        #print('Batch idx {}, data shape {}, target shape {}'.format(batch_idx, data.shape, target.shape))
        elbo = net.elbo(target.to(device),data.to(device))
        reg_loss = l2_regularisation(net._prior)+l2_regularisation(net._posterior)+l2_regularisation(net._f_comb)
        loss = -elbo + 1e-5*reg_loss
        score = batch_dice(F.softmax(net.sample(data,mean=False)))
        #running_loss += loss.item() * inputs.size(0) 
        #print(loss) 
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
        running_train_loss.append(loss.item())
        running_train_score.append(score.item())
        print('loss batch: {},score batch: {}, batch_idx: {}'.format(loss.item(),score.item(),batch_idx))
    else:
        running_val_loss=[]
        running_val_score=[]
          
        with torch.no_grad():
            for data,target in val_loader:
          
                elbo = net.elbo(target.to(device),data.to(device))
                reg_loss = l2_regularisation(net._prior)+l2_regularisation(net._posterior)+l2_regularisation(net._f_comb)
                loss = -elbo + 1e-5*reg_loss
                score = batch_dice(F.softmax(net.sample(data,mean=False)))
                running_train_loss.append(loss.item())
                running_train_score(score.item())
        
    epoch_train_loss,epoch_train_score = np.mean(running_train_loss),np.mean(running_train_score)
    print('Train loss : {} Dice score : {}'.format(epoch_train_loss,epoch_train_score)
    train_loss.append(epoch_train_loss)
    dice_score_train.append(epoch_train_score)
        
    epoch_val_loss,epoch_val_score = np.mean(running_val_loss),np.mean(running_val_score)
    print('Train loss : {} Dice score : {}'.format(epoch_val_loss,epoch_val_score)
    val_loss.append(epoch_val_loss)
    dice_score_val.append(epoch_val_score)
          
    checkpoint = { 'epoch': epoch +1,
                  'valid_loss_min':epoch_val_loss,
                  'state_dict':net.state_dict(),
                  'optimizer':optimizer.state_dict(),
        
    }
    save_ckp(checkpoint, False,checkpoint_path,best_model_path)
     
    if epoch_val_loss <= valid_loss_min:
          print('Validation loss decreased ({:.6f} =======> {:.6f}). Saving model ...'.format(valid_loss_min,epoch_val_loss))
          
          save_ckp(checkpoint, True,checkpoint_path,best_model_path)
          valid_loss_min=epoch_val_loss
          
    time_passed = time.time() - started
    print('{:.0f}m {:.0f}s'.format(time_passed//60, time_passed%60))