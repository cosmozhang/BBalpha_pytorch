
import numpy as np
import os, pickle, sys, time
import timeit

## import pytorch modules
import torch
from torch.autograd import Variable
import torch.nn.functional as Func
import torch.nn as nn
from torchvision import datasets, transforms

def log_sum_exp(tensor, dim  = None):
    xmax, _ = torch.max(tensor, dim = dim, keepdim = True)
    xmax_, _ = torch.max(tensor, dim = dim)
    return xmax_ + torch.log(torch.sum(torch.exp(tensor - xmax), dim = dim))

class MC_Loss(nn.Module):
    '''
    bbalpha softmax cross entropy with mc_logits
    '''

    def __init__(self, alpha= 1.0, k_mc = 20):
        super(MC_Loss, self).__init__()
        self.alpha = alpha
        if alpha == 0:
            print "alpha = 0"
        self.k_mc = k_mc

    def forward(self, mc_logits, y_true):
        len_y_true = len(y_true)
        y_true = y_true.expand(self.k_mc, -1, -1).contiguous().permute(1, 0, 2)
        if self.alpha != 0:
            # log(p_ij), p_ij = softmax(logit_ij)
            #assert mc_logits.ndim == 3
            temp, _ = torch.max(mc_logits, dim=2, keepdim=True)
            mc_log_softmax = mc_logits - temp
            mc_log_softmax = mc_log_softmax - torch.log(torch.sum(torch.exp(mc_log_softmax), dim=2, keepdim=True))
            mc_ll = torch.sum(y_true * mc_log_softmax, dim = -1)  # N x K
            # print mc_ll.size()
            out = - 1. / self.alpha * (log_sum_exp(self.alpha * mc_ll, 1) + np.log(1.0 / self.k_mc))
            # print out.size()
            # sys.exit()
            return torch.sum(out)
        else:
            predictions = Func.log_softmax(mc_logits, dim=2)
            # print y_true, predictions
            out = - torch.sum(torch.mean(y_true * predictions, dim=1))
            # print out.size()
            # sys.exit()
            return out

class BayesCNN(nn.Module):
    def __init__(self, alpha=1.0, k_mc=20, wd = 10**-6):
        super(BayesCNN, self).__init__()
        # input is 28x28
        # padding=2 for same padding
        self.wd = wd
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        # feature map size is 14*14 by pooling
        # padding=2 for same padding
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # feature map size is 7*7 by pooling
        self.fc1 = nn.Linear(64*14*14, 128)
        self.fc2 = nn.Linear(128, 10)
        self.k_mc = k_mc
        self.alpha = alpha
        self.bbalpha_loss = MC_Loss(self.alpha, self.k_mc)

    def lin(self, x):
        x = Func.relu(self.conv1.forward(x))
        # print x.size()
        x = Func.relu(self.conv2.forward(x))
        x = Func.max_pool2d(x, 2)
        x = Func.dropout(x, p = 0.5, training=self.training)
        x = x.view(-1, 64*14*14)   # reshape Variable
        x = Func.relu(self.fc1.forward(x))
        x = Func.dropout(x, p = 0.5, training=self.training)
        out = self.fc2.forward(x)
        return out

    def generate_MC_samples(self, x):
        # k_mc: number of samples
        if self.k_mc == 1:
            out = self.lin(x)
            mc_logits = out.view(len(out), 1, -1) # nb_batch x K_mc x nb_classes
            # print mc_logits.size()
            return mc_logits
        else:
            output_list = []
            for _ in xrange(self.k_mc):
                output_list += [self.lin(x)]  # THIS IS BAD!!! we create new dense layers at every call!!!!
            # print len(output_list)
            # sys.exit()
            output = torch.stack(output_list) # K_mc x nb_batch x nb_classes
            # print output.size()
            mc_logits = output.permute(1, 0, 2) # nb_batch x K_mc x nb_classes
            # print mc_logits.size()
            # sys.exit()
            return mc_logits

    def forward(self, x):
        lin_out = self.generate_MC_samples(x)
        out = Func.softmax(lin_out, dim=-1)
        # print out.size()
        out = torch.mean(out, dim=1).squeeze()
        return out

    def cal_bbalpha_loss(self, x, true_y): # negative log likelihood
        mc_logits = self.generate_MC_samples(x)
        loss = self.bbalpha_loss.forward(mc_logits, true_y)
        # negative log-likelihood
        return loss

    def cal_priors(self): # dropout may cause two forwards different
        prior = 0
        for param in self.parameters():
            prior = prior + self.wd * torch.sum(param**2)
        return prior

    def cal_npos(self, x, true_y): # negative log posterior
        return self.cal_bbalpha_loss(x, true_y) + self.cal_priors()



def main():

    """ Step 0: Compiling Model """
    # torch.manual_seed(123)
    # print bayescnn

    batch_size = 150
    num_epochs = 5

    bayescnn = BayesCNN(alpha = 0.5, k_mc = 3)
    optimizer = torch.optim.SGD(bayescnn.parameters(), lr=0.001)

    if torch.cuda.is_available():
        print "Run on GPU"
        bayescnn.cuda()
    else:
        print "Run on CPU"


    """ Step 1: Preprocessing """
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
        batch_size=100)

    # '''
    """ step 2: Training """
    bayescnn.train()
    start = timeit.default_timer()
    for epoch in xrange(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # print images.size()
            # print labels.size()
            images = Variable(images)
            d_labels = Variable(torch.zeros(len(images), 10).scatter_(1, labels.view(-1, 1), 1))
            # print labels, torch.max(d_labels, 1)[1]
            if torch.cuda.is_available():
                images = images.cuda()
                d_labels = d_labels.cuda()

            # Forward + Backward + Optimize
            bayescnn.zero_grad()
            #loss = bayescnn.cal_nllloss(images, labels)
            loss = bayescnn.cal_npos(images, d_labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print 'Epoch [{}/{}], Iter [{}/{}]'.format(epoch+1, num_epochs, i+1, len(train_loader))

    stop = timeit.default_timer()
    print("Time: ", stop - start )
    # '''

    """ Step 2: Model test """
    bayescnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images)
        # labels = Variable(labels)
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        outputs = bayescnn.forward(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print 'Test Accuracy of the model on the 10000 test images: {:0.2f}%'.format(100.0 * correct / total)

if __name__ == "__main__":
    main()
