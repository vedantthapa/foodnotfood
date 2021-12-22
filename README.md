# foodnotfood
A deep learning based web app to classify food and non food images

Live at - https://foodnotfood.herokuapp.com/
Front-end code - https://github.com/vedantthapa/foodnotfood-UI

## Dataset
The data was collected by extracting images from the [ImageNet](https://image-net.org/) and [Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) datasets.
- Food Images 
  - Random subset of Food101
- Non-Food Images 
  - 50 Images per class for 1000 classes.
  - Since the classes were randomly downloaded. Some of those classes contained food labels which were then manually moved to the appropriate class.
- Final distribution of the target:
  - Food Images ~ 66%
  - Non-Food Images ~ 34%

## References
- https://github.com/mf1024/ImageNet-Datasets-Downloader
- https://github.com/mrdbourke/food-not-food
