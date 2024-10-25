import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np
import polygnn_trainer as pt
from torch_geometric.loader import DataLoader

import polygnn.layers as layers

from torch_geometric.datasets import ZINC

random.seed(2)
torch.manual_seed(2)
np.random.seed(2)

# ##########################
# Multi-task models
# #########################
class polyGNN(pt.std_module.StandardModule):
    """
    Multi-task GNN model for polymers.
    """

    def __init__(
        self,
        node_size,
        edge_size,
        selector_dim,
        hps,
        normalize_embedding=True,
        graph_feats_dim=0,
        debug=False,
    ):
        """
        Initialize the PolyGNN model.

        Args:
            node_size (int): Size of the node features.
            edge_size (int): Size of the edge features.
            selector_dim (int): Dimension of the selector.
            hps (HpConfig): Hyperparameters object.
            normalize_embedding (bool, optional): Flag to normalize embeddings. Defaults to True.
            graph_feats_dim (int, optional): Dimension of the graph features. Defaults to 0.
            debug (bool, optional): Flag to enable debug mode. Defaults to False.
        """
        super().__init__(hps)
        self.node_size = node_size
        self.edge_size = edge_size
        self.selector_dim = selector_dim
        self.normalize_embedding = normalize_embedding
        assert isinstance(graph_feats_dim, int)
        self.graph_feats_dim = graph_feats_dim
        self.debug = debug

        self.mpnn = layers.MtConcat_PolyMpnn(
            node_size,
            edge_size,
            selector_dim,
            self.hps,
            normalize_embedding,
            debug,
        )

        self.final_mlp = pt.layers.Mlp(
            input_dim=self.mpnn.readout_dim + self.selector_dim + self.graph_feats_dim,
            output_dim=32,
            hps=self.hps,
            debug=False,
        )

        self.out_layer = pt.layers.my_output(size_in=32, size_out=1)

    def get_polymer_fps(self, data):
        """
        Get the polymer fingerprints by passing the data through the MPNN.

        Args:
            data (Data): Input data.

        Returns:
            tensor: Output fingerprint tensor.
        """
        return self.mpnn(data.x, data.edge_index, data.edge_weight, data.batch)

    def forward(self, data):
        """
        Forward pass of the model.

        Args:
            data (Data): Input data.

        Returns:
            tensor: Output tensor.
        """
        data.yhat = self.get_polymer_fps(data)
        data.yhat = F.leaky_relu(data.yhat)
        data.yhat = self.assemble_data(data)
        data.yhat = self.final_mlp(data.yhat)
        data.yhat = self.out_layer(data.yhat)
        return data.yhat.view(data.num_graphs, 1)


def ZINC_pretrain(model: polyGNN) -> polyGNN:
    zinc_dataset = ZINC(root = './data', split='train')
    zinc_dataset_val = ZINC(root = './data', split='val')   

    batch_size = 256
    train_loader = DataLoader(zinc_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(zinc_dataset_val, batch_size=batch_size, shuffle=True)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)

    def train(batch, step_optimizer=True):    
        model.train()
        optimizer.zero_grad()

        output = model(x=batch.x, 
                    edge_index=batch.edge_index, 
                    edge_attr=batch.edge_attr, 
                    batch_index=batch.batch)
        
        loss = loss_fn(output.squeeze(1), batch.y)

        loss.backward()
        if step_optimizer:
            optimizer.step()


        return loss.item()
    
    def validate(batch):
        model.eval()

        with torch.no_grad():
            output = model(batch)
            return loss_fn(output.squeeze(1), batch.y).item()
        
    def validation_step():
        loss_sum = 0
        for batch in val_loader:
            loss_sum += validate(batch)
        return loss_sum / len(val_loader)
    
    epochs = 100
    average_over = 100
    validate_every = 1000
    losses = []
    val_losses = []

    global_counter = 0


    for e in range(1, epochs+1):
        loss_sum = 0
        loss_count = 0

        for i, batch in enumerate(train_loader):

            train_loss = train(batch=batch, step_optimizer=True)

            loss_sum += train_loss
            loss_count += 1

            if i % average_over == average_over - 1:
                losses.append(loss_sum / loss_count)
                loss_sum = 0
                loss_count = 0

            if global_counter % validate_every == validate_every - 1:

                val_losses.append(validation_step())

            global_counter += 1
    
    return model
    

            
