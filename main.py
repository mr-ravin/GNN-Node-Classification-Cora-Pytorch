import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import os
import glob
from tqdm import tqdm
from gnn_model import GCN
from utils import save_plot, filtered_result


parser = argparse.ArgumentParser(description = "Graph Neural Networks for estimating water solubility of a molecule structure.")
parser.add_argument('-lr', '--learning_rate', default = 4e-3)
parser.add_argument('-ep', '--epoch', default = 2000)
parser.add_argument('-m', '--mode', default="train")
parser.add_argument('-g', '--num_graphs_per_batch', default=6)
args = parser.parse_args()

lr = args.learning_rate
total_epoch = int(args.epoch)
MODE = args.mode.lower()


dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
model = GCN(dataset, hidden_channels=24)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr) 
lr_scheduler = StepLR(optimizer, step_size=400, gamma=0.05)
criterion = nn.CrossEntropyLoss()
data = dataset[0] # dataset contains single graph, thus using index 0

def train():
      model.train()
      optimizer.zero_grad() 
      # Use all data as input, because all nodes have node features
      out = model(data.x, data.edge_index)  
      # Only use nodes with labels available for loss calculation --> mask
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  
      loss.backward() 
      optimizer.step()
      return loss.detach().numpy()


def test():
      model.eval()
      out = model(data.x, data.edge_index)
      # Use the class with highest probability.
      pred = out.argmax(dim=1)  
      # Check against ground-truth labels.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  
      # Derive ratio of correct predictions.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  
      return test_acc


if MODE == "train":
    losses = []
    for epoch in tqdm(range(1, total_epoch+1), desc= "Training Epoch"):
        loss = train()
        loss = float(loss)
        losses.append(loss)
        if epoch % 100 == 0:
            os.system("rm ./weights/*.pt")
            torch.save(model.state_dict(),"./weights/"+str(epoch)+".pt")
            print("Weight saved at epoch: ", epoch)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        lr_scheduler.step()
    losses, collect_at_each = filtered_result(losses)
    save_plot(losses, collect_at_each)
    test_acc = test()
    print(f'Test Accuracy: {test_acc:.4f}')

elif MODE == "test":
    weight_filename_list = glob.glob("./weights/*.pt")
    if len(weight_filename_list) !=1:
        print("Error: Weight file not present inside ./weights/")
        exit()
    else:
        weight_filename = weight_filename_list[0]
        print("Loading weight file for testing: ", weight_filename)
        model.load_state_dict(torch.load(weight_filename))
        test_acc = test()
        print(f'Test Accuracy: {test_acc:.4f}')