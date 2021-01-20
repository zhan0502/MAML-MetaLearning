import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader

from modelomniglotcnn import ConvolutionalNeuralNetwork
 
###################for linear model only############
#from modellinear import LinearNeuralNetwork
##################################################

from utils import update_parameters, get_accuracy

data_dir = 'omniglot'
dataset = omniglot(data_dir, shots=1, ways=5, shuffle=True, test_shots=15, meta_train=True, download=True) # for 1 shots
#dataset = omniglot(data_dir, shots=5, ways=5, shuffle=True, test_shots=15, meta_train=True, download=False) # for 5 shots

dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=0)

#CNN
model = ConvolutionalNeuralNetwork(1, 5, hidden_size=64)

#Linear
###################for linear model only############
#model = LinearNeuralNetwork(784, 5, hidden_size=64)
##################################################

model.to(device='cuda')
meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
device = 'cuda'
model.train()


with tqdm(dataloader, total=16) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()
            #print(batch_idx)
            train_inputs, train_targets = batch['train']
            ###################for linear model only############
            #print(train_inputs.shape)
            #train_inputs = train_inputs.reshape(16,5, 784) # for 1 shot
            #train_inputs = train_inputs.reshape(16,5*5, 784) # for 5 shots
            ##################################################
            
            train_inputs = train_inputs.to(device)
            train_targets = train_targets.to(device)

            test_inputs, test_targets = batch['test']
            ###################for linear model only############
            #print(test_inputs.shape)
            #test_inputs = test_inputs.reshape(16,75,784)
             ##################################################
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
                    step_size=0.04, first_order=True)

                test_logit = model(test_input, params=params)
                outer_loss += F.cross_entropy(test_logit, test_target)

                with torch.no_grad():
                    accuracy += get_accuracy(test_logit, test_target)

            outer_loss.div_(16)
            accuracy.div_(16)

            outer_loss.backward()
            meta_optimizer.step()

            pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))
            if batch_idx >= 2000:
                break