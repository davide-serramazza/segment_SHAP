from torch import concatenate as cat
from torch.utils.data import Dataset

class SHAP_dataloader(Dataset):

    # TODO assuming a static background

    def __init__(self, X, y,masks, background_dim):
        super().__init__()
        self.X = cat( X , axis=0)
        self.y = cat( y, axis=0)
        self.masks = cat( masks , axis=0)
        self.background_dim = background_dim

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.background_dim==1:
            return self.X[idx],  self.y[idx] , self.masks[idx]
        else:
            return self.X[idx].repeat(self.background_dim,1,1),  self.y[idx].repeat(self.background_dim) , self.masks[idx].repeat(self.background_dim,1,1)