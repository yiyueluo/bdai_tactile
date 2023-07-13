import torch.nn as nn
import time
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import io, os
import argparse
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler 
from model import tacNet
from dataloader import sample_data
# from sampler import custom_sampler
import pickle
import torch
import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from progressbar import ProgressBar
from torch.utils.tensorboard import SummaryWriter
# import shap

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--exp_dir', type=str, default='./dataset/01_bolt/', help='Experiment path')
parser.add_argument('--exp_name', type=str, default='_01_bolt', help='Experiment name')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size,128')
parser.add_argument('--weightdecay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--window', type=int, default=20, help='window around the time step')
parser.add_argument('--cls', type=int, default=6, help='number of class')
parser.add_argument('--epoch', type=int, default=500, help='The time steps you want to subsample the dataset to,500')
parser.add_argument('--ckpt', type=str, default='val_best_01_bolt.path', help='loaded ckpt file')
parser.add_argument('--eval', type=bool, default=False, help='Set true if eval time')
parser.add_argument('--train_continue', type=bool, default=False, help='Set true if eval time')
args = parser.parse_args()

if not os.path.exists(args.exp_dir + 'ckpts'):
    os.makedirs(args.exp_dir + 'ckpts')

if not os.path.exists(args.exp_dir + 'predictions'):
    os.makedirs(args.exp_dir + 'predictions')

w = [0.1790511,  0.13699978, 0.31752904, 0.07061999, 0.01554619, 0.2802539 ]

if not args.eval:
    train_path = args.exp_dir + 'train/'
    train_dataset = sample_data(train_path, args.window)
    w = pickle.load(open(args.exp_dir + 'train/sample_weight.p', "rb"))
    # print (len(w))
    train_sampler =  WeightedRandomSampler(w, len(train_dataset), replacement=False)
    # print (list(train_sampler.probs))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=8)
    print(len(train_dataset))

    val_path = args.exp_dir + 'val/'
    val_dataset = sample_data(val_path, args.window)
    w = pickle.load(open(args.exp_dir + 'val/sample_weight.p', "rb"))
    val_sampler =  WeightedRandomSampler(w, len(val_dataset), replacement=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=8)
    print(len(val_dataset))

if args.eval:
    test_path = args.exp_dir + 'test/'
    test_dataset = sample_data(test_path, args.window)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    print(len(test_dataset))

print(args.exp_dir, args.window)

'''training code'''
if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    # device = 'cuda:0'
    device = 'cpu'
    model = tacNet(args)  # model
    model.to(device)
    best_train_loss = np.inf
    best_val_loss = np.inf

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=5, verbose=True)
    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    # criterion = nn.BCELoss()


    if args.train_continue:
        checkpoint = torch.load(args.exp_dir + 'ckpts/' + args.ckpt + args.exp_name + '.path.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epochs = checkpoint['epoch']
        loss = checkpoint['loss']
        print("ckpt loaded", loss, "Now continue training")

    '''evaluation and optimize'''
    if args.eval:
        eval_loss = []
        tac_list = np.zeros((1, args.window, 9, 22))
        label_list = np.zeros((1, args.cls))
        pred_list = np.zeros((1, args.cls))
        checkpoint = torch.load(args.exp_dir + 'ckpts/' + args.ckpt + '.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epochs = checkpoint['epoch']
        loss = checkpoint['loss']
        print("ckpt loaded:", args.ckpt, loss, "Now running on eval set")
        model.eval()

        bar = ProgressBar(max_value=len(test_dataloader))
        for i_batch, sample_batched in bar(enumerate(test_dataloader, 0)):
            tac = torch.tensor(sample_batched[0], dtype=torch.float, device=device)
            label = torch.tensor(sample_batched[1], dtype=torch.float, device=device)
            label = torch.squeeze(label)

            with torch.set_grad_enabled(False):
                pred = model(tac)

            tac_list = np.concatenate((tac_list, tac.cpu().data.numpy()), axis=0)
            label_list = np.concatenate((label_list, label.cpu().data.numpy()), axis=0)
            pred_list = np.concatenate((pred_list, pred.cpu().data.numpy()), axis=0)

            loss = criterion(pred, label) * 10 

            eval_loss.append(loss.data.item())

        pickle.dump([tac_list, label_list, pred_list],
                    open(args.exp_dir + 'predictions/eval' + args.exp_name +'.p', "wb"))

        print ('loss:', np.mean(eval_loss))

    else:
        writer = SummaryWriter(comment=args.exp_name)
        n = 0

        for epoch in range(args.epoch):
            train_loss = []
            val_loss = []

            bar = ProgressBar(max_value=len(train_dataloader))

            for i_batch, sample_batched in bar(enumerate(train_dataloader, 0)):
                model.train(True)
                tac = torch.tensor(sample_batched[0], dtype=torch.float, device=device)
                label = torch.tensor(sample_batched[1], dtype=torch.float, device=device)
                label = torch.squeeze(label)

                # print (np.mean(label.cpu().data.numpy()), np.amax(label.cpu().data.numpy()))
                # print(act.size(), tactile.size())

                with torch.set_grad_enabled(True):
                    pred = model(tac)
                
                # print (pred.size(), label.size())
                loss = criterion(pred, label) * 10 
                # print (loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss.append(loss.data.item())

                writer.add_scalar('Loss/train_perBatch', loss, epoch * len(train_dataloader) + i_batch)
                writer.add_scalar('Loss/train_meanSoFar', np.mean(train_loss),
                                  epoch * len(train_dataloader) + i_batch)

                pickle.dump([tac.cpu().data.numpy(), label.cpu().data.numpy(), pred.cpu().data.numpy()],
                            open(args.exp_dir + 'predictions/train' + args.exp_name + '.p', "wb"))

                if i_batch % 20 == 0 and i_batch != 0:
                    val_loss_t = []
                    n += 1

                    print("[%d/%d], Loss: %.6f" % (
                        i_batch, len(train_dataloader),loss.item()))

                    # torch.save({
                    #     'epoch': epoch,
                    #     'model_state_dict': model.state_dict(),
                    #     'optimizer_state_dict': optimizer.state_dict(),
                    #     'loss': loss, },
                    #     args.exp_dir + 'ckpts/train_online_' + str(args.window) + '_' + str(epoch) + args.exp_name + '.path.tar')

                    print("Now running on val set")
                    model.train(False)

                    bar = ProgressBar(max_value=len(val_dataloader))
                    for i_batch, sample_batched in bar(enumerate(val_dataloader, 0)):
                        tac = torch.tensor(sample_batched[0], dtype=torch.float, device=device)
                        label = torch.tensor(sample_batched[1], dtype=torch.float, device=device)
                        label = torch.squeeze(label)

                        with torch.set_grad_enabled(False):
                            pred = model(tac)

                        # print (pred.size(), label.size())
                        loss = criterion(pred, label) * 10 

                        
                        # print (np.mean(label.cpu().data.numpy()), np.amax(label.cpu().data.numpy()))

                        if i_batch % 100 == 0 and i_batch != 0:
                            print("[%d/%d], Loss: %.6f, min: %.6f, mean: %.6f, max: %.6f"% (
                                i_batch, len(train_dataloader), loss.item(), np.amin(pred.cpu().data.numpy()),
                                np.mean(pred.cpu().data.numpy()), np.amax(pred.cpu().data.numpy())))

                        val_loss.append(loss.data.item())
                        val_loss_t.append(loss.data.item())
                    writer.add_scalar('Loss/val', np.mean(val_loss_t), n)

                    # scheduler.step(np.mean(val_loss))
                    if np.mean(train_loss) < best_train_loss:
                        print("new best train loss:", np.mean(train_loss))
                        best_train_loss = np.mean(train_loss)

                        pickle.dump([tac.cpu().data.numpy(), label.cpu().data.numpy(), pred.cpu().data.numpy()],
                                    open(args.exp_dir + 'predictions/train_best' + args.exp_name + '.p', "wb"))


                    # print ("val_loss:", np.mean(val_loss))
                    if np.mean(val_loss_t) < best_val_loss:
                        print("new best val loss:", np.mean(val_loss_t))
                        best_val_loss = np.mean(val_loss_t)

                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss, },
                            args.exp_dir + 'ckpts/val_best' + args.exp_name + '.path.tar')

                avg_train_loss = np.mean(train_loss)
                avg_val_loss = np.mean(val_loss)

                # avg_train_loss = np.array([avg_train_loss])
                # avg_val_loss = np.array([avg_val_loss])
                #
                # train_loss_list = np.append(train_loss_list, avg_train_loss, axis=0)
                # val_loss_list = np.append(val_loss_list, avg_val_loss, axis=0)

            print("Train Loss: %.6f, Valid Loss: %.6f" % (avg_train_loss, avg_val_loss))

            
            # appply SHAP
            # batch = next(iter(train_dataloader))
            # # print (len(batch))
            # img, label = batch
            # # print (label.shape)
            # img = torch.tensor(img, dtype=torch.float, device=device)
            # print (np.argmax(label[100:105, 0, :], axis=1))

            # background = img[:100]
            # test_images = img[100:105]

            # e = shap.DeepExplainer(model, background)
            # shap_values = e.shap_values(test_images)
            # shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
            # test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
            # shap.image_plot(shap_numpy, -test_numpy)