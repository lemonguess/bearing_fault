import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DiffusionConv, global_mean_pool
from torch_geometric.data import Data
from torch.optim import Adam

class DCRNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes):
        super(DCRNN, self).__init__()
        self.num_layers = num_layers
        self.diffusion_convs = nn.ModuleList()
        self.lstms = nn.ModuleList()

        # Diffusion convolution layers
        for _ in range(num_layers):
            self.diffusion_convs.append(DiffusionConv(in_channels, hidden_channels))

        # LSTM layers
        for _ in range(num_layers):
            self.lstms.append(nn.LSTM(hidden_channels, hidden_channels))

        # Output layer
        self.output_layer = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Diffusion convolution
        for i in range(self.num_layers):
            x = F.relu(self.diffusion_convs[i](x, edge_index))

        # LSTM
        for i in range(self.num_layers):
            x = x.view(-1, self.num_nodes, self.hidden_channels)
            x, _ = self.lstms[i](x)

        # Global pooling
        x = global_mean_pool(x, data.batch)

        # Output layer
        x = self.output_layer(x)

        return F.log_softmax(x, dim=1)

# Example usage
num_nodes = 10
in_channels = 3
hidden_channels = 64
out_channels = 2
num_layers = 2

# Generate random graph data (replace with your actual data)
x = torch.randn(num_nodes, in_channels)
edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                           [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)
data = Data(x=x, edge_index=edge_index)

# Create DCRNN model
model = DCRNN(in_channels, hidden_channels, out_channels, num_layers, num_nodes)

# Forward pass
output = model(data)
print(output)

# Define loss function and optimizer
criterion = nn.NLLLoss()
optimizer = Adam(model.parameters(), lr=0.01)

# Example training loop (replace with your actual training loop)
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
