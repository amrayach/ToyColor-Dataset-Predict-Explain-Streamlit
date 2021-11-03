from torch import nn
import torch


class MLP(nn.Module):
    def __init__(self, args):

        super(MLP, self).__init__()
        self.args = args
        self.layers = nn.Sequential(
            #nn.Flatten(),
            nn.Linear(args.getint('Model', 'input_dim'), 50),
            nn.ReLU(),
            nn.Linear(50, 30),
            nn.ReLU(),
            nn.Linear(30, args.getint('Model', 'output_dim'))
        )

    def forward(self, x):
        #y = x
        x = x.view(x.size(0), -1)
        #yy = x.view(-1, x.size(0))
        #yyy = x.view(x.size(0), 3, 5, 5)
        #a = torch.eq(y, yyy)
        #b = torch.all(y.eq(yyy))
        #x = self.layers(x)

        return self.layers(x)

    def get_optimizer(self):
        if self.args.get('Train', 'optimizer') == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.getfloat('Train', 'learning_rate'),
                                         weight_decay=self.args.getfloat('Train', 'weight_decay'))
        else:
            raise Exception("Please select a valid optimizer !!")

        return optimizer

    def get_loss_function(self):
        # Setup loss function
        if self.args.get('Train', 'loss_function') == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
        else:
            raise Exception("Please select a valid loss function !!")

        return criterion

    def get_device(self):
        return torch.device('cpu') if not self.args.getboolean('Device', 'enable_gpu') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')



