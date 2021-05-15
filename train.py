import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

from pointer_network import PtrNet
from neural_comb_opt_rl import NeuralCombOptRL, reward_tsp
from data_set import TSPDataset
from utils import plot_loss

def argparser():
    parser = ArgumentParser()
    parser.add_argument('-f', default='', type=str, help='configure file path')
    parser.add_argument('--name', default='PtrNet', type=str, help='network name')
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

    # Training NeuralCombOptRL
    parser.add_argument('--actor_net_lr', default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--critic_net_lr', default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--actor_lr_decay_step', default=5000, help='')
    parser.add_argument('--critic_lr_decay_step', default=5000, help='')
    parser.add_argument('--actor_lr_decay_rate', default=0.96, help='')
    parser.add_argument('--critic_lr_decay_rate', default=0.96, help='')
    parser.add_argument('--reward_scale', default=2, type=float,  help='')
    parser.add_argument('--is_train', type=bool, default=True, help='')
    parser.add_argument('--random_seed', default=24601, help='')
    parser.add_argument('--max_grad_norm', default=2.0, help='Gradient clipping')
    parser.add_argument('--critic_beta', type=float, default=0.9, help='Exp mvg average decay')
    
    return parser.parse_args()

def construct(model_name, params, num_workers=0, USE_CUDA=False):
    solve_exactly = True
    if model_name == "PtrNet":
        model = PtrNet(params.embedding_dim, params.hidden_dim, params.num_lstms, params.dropout, params.bidir)
    elif model_name == "NeuralCombOptRL":
        model = NeuralCombOptRL(
            params.ncity, params.embedding_dim, params.hidden_dim, params.num_lstms,
            params.dropout, reward_tsp, bidirectional=params.bidir, is_train=True, use_cuda=USE_CUDA)
        solve_exactly = False
    else:
        raise NotImplementedError

    dataset = TSPDataset(params.train_size, params.ncity, solve=solve_exactly)

    dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=num_workers)
        
    if USE_CUDA:
        num_gpu = torch.cuda.device_count()
        print(f"Using GPU {num_gpu} devices.")
        model.cuda()
        net = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    return model, dataset, dataloader

def train_PtrNet(params, num_workers=0):
    USE_CUDA = bool(params.gpu and torch.cuda.is_available())
    model, dataset, dataloader = construct("PtrNet", params, num_workers=num_workers, USE_CUDA=USE_CUDA)

    CCE = torch.nn.CrossEntropyLoss()
    model_optim = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.lr)
    
    losses = []
    model.train()
    for epoch in range(params.n_epoch):
        batch_loss = []
        iterator = tqdm(dataloader, unit="Batch")

        for batch_i, sample in enumerate(iterator):
            iterator.set_description(f"Batch {batch_i+1}/{params.n_epoch}")

            train_batch = Variable(sample["coordinate"])
            target_batch = Variable(sample["solution"])

            if USE_CUDA:
                train_batch = train_batch.cuda()
                target_batch = target_batch.cuda()
            output, p = model(train_batch)

            output = output.contiguous().view(-1, output.size()[-1])

            target_batch = target_batch.view(-1)

            loss = CCE(output, target_batch)

            losses.append(loss.item())
            batch_loss.append(loss.item())

            model_optim.zero_grad()
            loss.backward()
            model_optim.step()

            iterator.set_postfix(loss=f"{loss.item()}")
        iterator.set_postfix(loss=np.average(batch_loss))
    
    return losses

def train_NeuralCombOptRL(params, num_workers=0):
    USE_CUDA = bool(params.gpu and torch.cuda.is_available())
    model, dataset, dataloader = construct("NeuralCombOptRL", params, num_workers=num_workers, USE_CUDA=USE_CUDA)

    actor_optim = optim.Adam(filter(lambda p: p.requires_grad, model.actor_net.parameters()), lr=params.lr)
    actor_scheduler = lr_scheduler.MultiStepLR(actor_optim,
        range(params.actor_lr_decay_step, params.actor_lr_decay_step * 1000,
            params.actor_lr_decay_step), gamma=params.actor_lr_decay_rate)
    critic_exp_mvg_avg = torch.zeros(1)

    losses = []
    model.train()
    for epoch in range(params.n_epoch):
        batch_loss = []
        iterator = tqdm(dataloader, unit="Batch")

        for batch_i, sample in enumerate(iterator):
            iterator.set_description(f"Batch {batch_i+1}/{params.n_epoch}")

            train_batch = Variable(sample["coordinate"])

            if USE_CUDA:
                train_batch = train_batch.cuda()

            R, probs, actions, action_idxs = model(train_batch)

            if batch_i == 0:
                critic_exp_mvg_avg = R.mean()
            else:
                critic_exp_mvg_avg = (critic_exp_mvg_avg * beta) + ((1. - beta) * R.mean())
            
            advantage = R - critic_exp_mvg_avg
            logprobs = 0
            nll = 0
            for prob in probs:
                logprob = torch.log(prob)
                nll += -logprob
                logprobs += logprob
            nll[(nll != nll).detach()] = 0.
            logprobs[(logprobs < -1000).detach()] = 0.

            reinforce = advantage*logprobs
            actor_loss = reinforce.mean()

            actor_optim.zero_grad()
            actor_loss.backward()
            
            torch.nn.utils.clip_grad_norm(model.actor_net.parameters(), params.max_grad_norm, norm_type=2)
            actor_optim.step()
            actor_scheduler.step()
            critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

            losses.append(actor_loss.item())
            batch_loss.append(actor_loss.item())

            iterator.set_postfix(loss=f"{loss.item()}")
        iterator.set_postfix(loss=np.average(batch_loss))
    
    return losses

if __name__=="__main__":
    # TODO: 
    # configファイルをyaml形式で作成して、configファイルが指定された時はそちらを使うようにする
    params = argparser()
    if params.name == "PtrNet":
        losses = train_PtrNet(params)
    else:
        losses = train_NeuralCombOptRL(params)
     
    breakpoint()
    fig, ax = plot_loss(losses)
    fig.show()

    breakpoint()