from torch.utils.data import Dataset
import cv2


class LaneDataset(Dataset):
    def __init__(self, imagePath, maskPath, transforms=None):
        self.imagePath = imagePath  # Array of filepaths for the input images
        self.maskPath = maskPath  # Array of filepaths for the mask images
        self.transforms = transforms

    def __len__(self):
        return len(self.imagePath)

    def __getitem__(self, idx):
        img_path = self.imagePath[idx]
        mask_path = self.maskPath[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transforms != None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        mask_binary = (mask > 0).type(torch.float)

        return (image, mask_binary)
