import torch
import torch.nn as nn

class SWFlowModel(nn.Module):
    """Swliced-Wasserstein Normalizing flow class

    Args:
        - flows (list of flows): list of flows to be applied in the model
        - device (str): device to use for the model

    Attributes:
        - forward (function): forward pass of the model
        - inverse (function): inverse pass of the model
        - transport_cost (function): transport cost of the model
        - sample_x (function): sample x from y sampler
        - sample_y (function): sample y from x sampler
        - forward_barycenter (function): forward pass until the nth flow
        - inverse_barycenter (function): inverse pass until the nth flow
    """
    def __init__(self, flows, device="cpu"):
        super().__init__()
        self.device = device
        self.flows = nn.ModuleList(flows).to(self.device)
        self.nb_flows = len(flows)

    def forward(self, x):
        m, _ = x.shape
        shatten = 0
        log_det = torch.zeros(m, device=self.device)
        for flow in self.flows:
            x, log_diag, ld = flow.forward(x)
            shatten += torch.sum(torch.pow(log_diag, 2))
            log_det += ld
        return x, shatten, log_det
    
    def transport_cost(self, x):
        cost = torch.zeros(self.nb_flows, device=self.device)
        for i, flow in enumerate(self.flows):
            z, _, _ = flow.forward(x)
            cost[i] = torch.norm(x - z, p='fro')
            x = z
        return cost
    
    def forward_barycenter(self, x, nb_flows):
        for flow in self.flows[:nb_flows]:
            x, _, _ = flow.forward(x)
        return x

    def inverse(self, z):
        m, _ = z.shape
        for flow in self.flows[::-1]:
            z, _ = flow.inverse(z)
        return z
    
    def inverse_barycenter(self, x, nb_flows):
        for flow in self.flows[::-1][:nb_flows]:
            x, _ = flow.inverse(x)
        return x
    
    def sample_x(self, y_sampler, nb_samples):
        data, label = y_sampler(nb_samples)
        y = torch.from_numpy(data.astype(np.float32)).to(self.device)
        x, _ = self.inverse(y)
        return x
    
    def sample_y(self, x_sampler, nb_samples):
        x, _ = x_sampler(nb_samples)
        data, label = x_sampler(nb_samples)
        x = torch.from_numpy(data.astype(np.float32)).to(self.device)
        y, _ = self.forward(x)
        return y