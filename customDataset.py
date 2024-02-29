import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

# needs a folder of images
# needs a csv file of imagenames in col1 and corresponding class in col2
class customDataet(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0   ])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)
        
        return (image, y_label)
    
#Example usage:
# dataset = customDataset(csv_file = "...", root_fir = "...", transform = transforms.toTensor())
# train_data, test_data = torch.utils.data.random_split(dataset, [5000, 2000])
# train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)