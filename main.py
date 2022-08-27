from dataset.dataset import Dataset
from dataset.dataloader import DataLoader, unpad
from utils import display_bounding_boxes, tensor_to_bounding_box, tensor_to_np_image


if __name__ == '__main__':
    # how to iterate through dataset
    path = r'data/train'
    dataset = Dataset(path, 64)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=32,
    )
    for batch, lengths in data_loader:
        images, bbs, numbers = batch
        bbs = unpad(bbs, lengths)
        numbers = unpad(numbers, lengths)

        # do stuff ...

        # displaying bounding boxes
        bbs = [tensor_to_bounding_box(bb) for bb in bbs[0]]
        image = tensor_to_np_image(images[0])
        display_bounding_boxes(image, bbs)

        break
