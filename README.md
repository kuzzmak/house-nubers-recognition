# Recognition of the street view house numbers with CNN

## Dataset

For the purposes of this project [SVHN](http://ufldl.stanford.edu/housenumbers/) (Street View House Numbers) [1] dataset will be used.

There are two variants:

- with normal images (all digits are on the same image)
- every digit is on the separate image

The first version will be used.

Furthermore, these version have _train_, _test_ and _extra_ datasets. Here, we will be only interested in the first two for the purpose of simplicity.

### How to structure dataset

- download _train_ and _test_ datasets from the link above
- make directory called _data_ in the root directory of the project
- extract downloaded images in a way that the following directory structure is made:
  - data/train/...images
  - data/test/...images

## Developing environment

1. make a virtual environment with

   ```bash
   python -m venv env
   ```

2. activate this environment

   ```bash
   env\Scripts\activate (windows)
   ```

3. install project requirements

   ```bash
   pip install -r requirements.txt
   ```

4. install [PyTorch and Torchvision](https://pytorch.org/) depending on the type of hardware you have

5. install yolov5 requirements

   ```bash
   pip install -r yolov5/requirements.txt
   ```

## Bounding box model

Model used for bounding boxes detection is [YOLOV5](https://github.com/ultralytics/yolov5). Only one class is used for the detection because we only have one type of images, house numbers images.

### 1 .Training

1. Prepare dataset for _yolo_ model by running following command:

   ```bash
   python make_svhnbb_dataset.py -image_shape 150 150 -data_size 10000
   ```

   After this command finishes, in directory _datasets_, _SVHNBB_ directory is created and this directory contains _images_ and _labels_ directories with _data.yaml_ file. Both, _images_ and _labels_ directories contain training and validation images and labels.

2. Run following command:

   ```bash
   python train_yolov5.py --data datasets\SVHNBB\data.yaml --hyp yolov5\data\hyps\hyp.scratch.yaml --img 150 --epochs 150 --batch-size 256 --workers 2
   ```

   When this process finishes, weights of the model are stored in _runs/train/exp/weights_ directory.

### 2. Inference

File _detect_digits.py_ shows how to user yolo model in order to make inference on some image.

## Resources

> [1] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng Reading Digits in Natural Images with Unsupervised Feature Learning NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011. </cite> [PDF](http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf)
