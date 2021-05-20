import torch
import torch.nn as nn
import torch.nn.functional as F

def _euclidean_distance(x, y=None):
    x_norm = torch.sum(torch.pow(x, 2), 1).view(-1, 1)
    if y is not None:
        y_norm = torch.sum(torch.pow(y, 2), 1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.t(y))
    return dist

class WeightedSoftMarginTripletLoss(nn.Module):
    def __init__(self, alpha=5.0, hardest=False):
        super(WightedSoftMarginTripletLoss, self).__init__()
        self.alpha = alpha
        self.hardest = hardest

    def forward(self, img, point_pos):
        batch_size = img.shape[0]
        dist = _euclidean_distance(img, point_pos)
        diag = torch.arange(0, batch_size)
        diag = diag.to(img.device)
        d_pos = dist[diag, diag].to(img.device)
        
        if self.hardest:
            dist[diag, diag] = float("Inf")
            d_neg, _ = torch.min(dist, 1).to(img.device)
        else:
            rand_indx = torch.randint(0, batch_size, (batch_size,), device=img.device)
            d_neg = dist.gather(1, rand_indx.view(-1,1)).flatten().to(img.device)
        
        d = d_pos - d_neg
        loss = torch.log(1 + torch.exp(self.alpha*d))
        loss = torch.mean(loss)     
        return loss
