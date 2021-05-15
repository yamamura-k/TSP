import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

from pointer_network import PtrNet
from data_set import TSPDataset
from utils import plot_loss

def argparser():
    parser = ArgumentParser()
    parser.add_argument('-f', default='', type=str, help='configure file path')
    # Data
    parser.add_argument('--train_size', default=1000000, type=int, help='Training data size')
    parser.add_argument('--val_size', default=10000, type=int, help='Validation data size')
    parser.add_argument('--test_size', default=10000, type=int, help='Test data size')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    # Train
    parser.add_argument('--n_epoch', default=50000, type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    # GPU
    parser.add_argument('--gpu', default=False, action='store_true', help='Enable gpu')
    # TSP
    parser.add_argument('--ncity', type=int, default=5, help='Number of points in TSP')
    # Network
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding size')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--num_lstms', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout value')
    parser.add_argument('--bidir', default=True, action='store_true', help='Bidirectional')
    
    return parser.parse_args()

def construct(model_name, params, num_workers=1, USE_CUDA=False):
    if model_name != "PtrNet":
        raise NotImplementedError
    model = PtrNet(params.embedding_dim, params.hidden_dim, params.num_lstms, params.dropout, params.bidir)

    dataset = TSPDataset(params.train_size, params.ncity)

    dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=num_workers)
        
    if USE_CUDA:
        num_gpu = torch.cuda.device_count()
        print(f"Using GPU {num_gpu} devices.")
        model.cuda()
        net = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    
    return model, dataset, dataloader

def train(model_name, params, num_workers=1):
    USE_CUDA = bool(params.gpu and torch.cuda.is_available())
    model, dataset, dataloader = construct(model_name, params, num_workers=num_workers, USE_CUDA=USE_CUDA)

    CCE = torch.nn.CrossEntropyLoss()
    model_optim = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.lr)
    
    losses = []
    for epoch in range(params.n_epoch):
        batch_loss = []
        iterator = tqdm(dataloader, unit="Batch")

        for batch_i, sample in enumerate(iterator):
            iterator.set_description(f"Batch {i+1}/{params.n_epoch}")

            train_batch = Variable(sample["coordinate"])
            target_batch = Variable(sample["solution"])

            if USE_CUDA:
                train_batch = train_batch.cuda()
                target_batch = target_batch.cuda()
            output, p = model(train_batch)

            output = output.contiguous().view(-1, output.size()[-1])

            target_batch = target_batch.view(-1)

            loss = CCE(output, target_batch)

            losses.append(loss.data[0])
            batch_loss.append(loss.data[0])

            model_optim.zero_grad()
            loss.backward()
            model_optim.step()

            iterator.set_postfix(loss=f"{loss.data[0]}")
        iterator.set_postfix(loss=np.average(batch_loss))
    
    return losses

if __name__=="__main__":
    name = "PtrNet"
    params = argparser()
    losses = train(name, params)
    # fig, ax = plot_loss(losses)
    # fig.show()