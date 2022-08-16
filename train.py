import os
import torch
import torchvision
from torch.utils.data import DataLoader
from d2l import torch as d2l
import torch.nn.functional as F

import config
import dataset
import model
import engine

def loss_fn(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1) 

def run_training():
    voc_train = dataset.VOCSegDataset(True, config.CROP_SIZE, config.DATA_DIR)
    voc_test = dataset.VOCSegDataset(False, config.CROP_SIZE, config.DATA_DIR)

    train_loader = DataLoader(voc_train, batch_size = config.BATCH_SIZE,num_workers = config.NUM_WORKERS, shuffle = True, drop_last=True)

    test_loader = DataLoader(voc_test, batch_size = config.BATCH_SIZE,num_workers = config.NUM_WORKERS, shuffle = False, drop_last=True)

    # Build Model

    net = model.ImageSegmentation(num_classes=config.NUM_CLASSES)
    # optimizer = torch.optim.Adam(net.parameters(),lr = 3e-4)
    # schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, factor = 0.8, patience =5, verbose = True)

    # checkpoint = torch.load("weights/best.pt")
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # net.to(config.DEVICE)

    num_epochs, lr, wd, devices = config.EPOCHS, 0.001, 1e-3, d2l.try_all_gpus()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
    engine.train_fn(net, train_loader, test_loader, loss_fn, optimizer, num_epochs, devices)

    # for epoch in range(config.EPOCHS):
    #     train_loss = engine.train_fn(net, train_loader, optimizer)
    #     valid_preds, valid_loss = engine.eval_fn(net, test_loader)

    #     valid_cap_preds = []

        # for vp in valid_preds:
        #     current_preds = decode_predictions(vp, lbl_encoder)
        #     valid_cap_preds.extend(current_preds)

        #df = pd.DataFrame(list(zip(test_orig_targets, valid_cap_preds)))
        #df.to_csv(f'preds/preds_{epoch}.csv',encoding='utf-8-sig', index = False)
        
        # print(list(zip(test_orig_targets, valid_cap_preds))[6:11])

        # print(f"Epoch: {epoch}, Train Loss: {train_loss}, Validation loss: {valid_loss}")

        # if min_loss > train_loss:
        #     min_loss = train_loss
        #     torch.save({ 'epoch': epoch,
        #         'model_state_dict': net.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': min_loss,}, "weights/best.pt")

if __name__ == "__main__":
    run_training()