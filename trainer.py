from utils import TSPDataset, reward
from tsp_rl import NCombOptRL
import glob
import os
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from torch import optim
from torch.optim import lr_scheduler
def test():
    train_fnames = [x for x in glob.glob("./ALL_tsp/*.tsp") if "100" in x and "1000" not in x and "2" not in x]
    tot = len(train_fnames)
    input_dim = 2
    emb_dim = 4
    hidden_dim = 4
    size = 4
    n_glimpses = 1
    n_process_blocks = 1
    C = 10
    tanh_flg=True
    reward_fn = reward
    training_dataset = TSPDataset(train=True, size=size, num_samples=4, dataset_fnames=train_fnames[tot//4:])
    val_dataset = TSPDataset(train=True, size=size, num_samples=4, dataset_fnames=train_fnames[:tot//4])
    # Instantiate the Neural Combinatorial Opt with RL module
    model = NCombOptRL(
        input_dim,
        emb_dim,
        hidden_dim,
        size, # decoder len
        n_glimpses,
        n_process_blocks,
        C,
        tanh_flg,
        reward,
        False)
    save_dir = "./tmp" 
    
    try:
        os.makedirs(save_dir)
    except:
        pass
    
    actor_optim = optim.Adam(model.actor.parameters(), lr=0.001)
    
    actor_scheduler = lr_scheduler.MultiStepLR(actor_optim, [i*10 for i in range(1, 10000)])
    
    training_dataloader = DataLoader(training_dataset, batch_size=3, shuffle=True, num_workers=1)
    
    validation_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1)
    
    critic_exp_mvg_avg = torch.zeros(1)
    beta = 0.8
    step = 0
    val_step = 0
    epoch = 0
    for i in range(epoch, epoch + 50):
        model.train()
        # sample_batch is [batch_size x input_dim x sourceL]
        for batch_id, sample_batch in enumerate(tqdm(training_dataloader, disable=False)):
            bat = Variable(sample_batch)
            R, probs, actions, actions_idxs = model(bat)
        
            if batch_id == 0:
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
           
            # guard against nan
            nll[(nll != nll).detach()] = 0.
            # clamp any -inf's to 0 to throw away this tour
            logprobs[(logprobs < -1000).detach()] = 0.

            # multiply each time step by the advanrate
            for i in range(len(logprobs)):
                logprobs[i] *= advantage[i]
            reinforce = logprobs
            print(reinforce)

            #reinforce = advantage * logprobs
            actor_loss = reinforce.mean()
            
            actor_optim.zero_grad()
           
            actor_loss.backward()

            # clip gradient norms
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0, norm_type=2)

            actor_optim.step()
            actor_scheduler.step()

            critic_exp_mvg_avg = critic_exp_mvg_avg.detach()
            
            step += 1
        
            if step % 50 == 0:
                print('epoch: {}, train_batch_id: {}, avg_reward: {}'.format(
                    i, batch_id, R.mean().data[0]))
                example_output = []
                example_input = []
                for idx, action in enumerate(actions):
                    if task[0] == 'tsp':
                        example_output.append(actions_idxs[idx][0].data[0])
                    else:
                        example_output.append(action[0].data[0])  # <-- ?? 
                    example_input.append(sample_batch[0, :, idx][0])
                print('Example train output: {}'.format(example_output))
    
        
        print('\n~Validating~\n')
    
        example_input = []
        example_output = []
        avg_reward = []
    
        # put in test mode!
        model.eval()
    
        for batch_id, val_batch in enumerate(tqdm(validation_dataloader,
                disable=False)):
            bat = Variable(val_batch)
    
            R, probs, actions, action_idxs = model(bat)
            
            avg_reward.append(R[0].data[0])
            val_step += 1.
    
            log_value('val_avg_reward', R[0].data[0], int(val_step))
    
            if val_step % int(args['log_step']) == 0:
                example_output = []
                example_input = []
                for idx, action in enumerate(actions):
                    if task[0] == 'tsp':
                        example_output.append(action_idxs[idx][0].data[0])
                    else:
                        example_output.append(action[0].data[0])
                    example_input.append(bat[0, :, idx].data[0])
                print('Step: {}'.format(batch_id))
                print('Example test output: {}'.format(example_output))
                print('Example test reward: {}'.format(R[0].data[0]))
        
        print('Validation overall avg_reward: {}'.format(np.mean(avg_reward)))
        print('Validation overall reward var: {}'.format(np.var(avg_reward)))
             
        print('Saving model...')
        
        torch.save(model, os.path.join(save_dir, 'epoch-{}.pt'.format(i)))

        training_dataset = tsp_task.TSPDataset(train=True, size=size,
            num_samples=100)
        training_dataloader = DataLoader(training_dataset, batch_size=1,
            shuffle=True, num_workers=1)

if __name__=="__main__":
    test()