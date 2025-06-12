from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from src.utils.image_utils import img_to_tensor

class DiffusionDataset(Dataset):
    def __init__(self, dataset_name='cifar10', split='train', image_size=32):
        """Initialize dataset for diffusion model training."""
        self.dataset = load_dataset(dataset_name)[split]
        self.image_size = image_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['img']
        if self.image_size != image.size[0]:
            image = image.resize((self.image_size, self.image_size))
        return img_to_tensor(image)

def get_dataloader(dataset_name='cifar10', batch_size=32, image_size=32, num_workers=4):
    """Create a DataLoader for the specified dataset."""
    dataset = DiffusionDataset(dataset_name=dataset_name, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    ) 