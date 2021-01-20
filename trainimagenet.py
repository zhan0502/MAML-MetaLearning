import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torchmeta.datasets.helpers import miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader

from modelimagenetcnn import ConvolutionalNeuralNetwork
from utils import update_parameters, get_accuracy

data_dir = 'miniimagenet' 
dataset = miniimagenet(data_dir, shots=1, ways=5, shuffle=True, test_shots=15, meta_train=True, download=True) #data 

dataloader = BatchMetaDataLoader(dataset, batch_size=8, num_workers=0)

model = ConvolutionalNeuralNetwork(3, 5, hidden_size=32)
model.to(device='cuda')
meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

device = 'cuda'
print(len(dataset))
model.train()

with tqdm(dataloader, total=8) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()
            #print(batch_idx)
            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device)
            train_targets = train_targets.to(device)
             
            #print(train_inputs.shape)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device)
            test_targets = test_targets.to(device)
            
             
            outer_loss = torch.tensor(0., device='cuda')
            accuracy = torch.tensor(0., device='cuda')
            for task_idx, (train_input, train_target, test_input,
                    test_target) in enumerate(zip(train_inputs, train_targets,
                    test_inputs, test_targets)):
                train_logit = model(train_input)
                inner_loss = F.cross_entropy(train_logit, train_target)

                model.zero_grad()
                params = update_parameters(model, inner_loss,
                    step_size=0.01, first_order=True)

                test_logit = model(test_input, params=params)
                outer_loss += F.cross_entropy(test_logit, test_target)

                with torch.no_grad():
                    accuracy += get_accuracy(test_logit, test_target)

            outer_loss.div_(8)
            accuracy.div_(8)

            outer_loss.backward()
            meta_optimizer.step()

            pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))
            if batch_idx >= 2000:
                break