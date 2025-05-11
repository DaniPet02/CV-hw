# CV-hw
Uncertainty-Aware Road Obstacle Identification


## The Task

The aim of this project is to develop a general, model-agnostic framework for road obstacle identification, starting from the outputs of any semantic segmentation network. The system will focus on anomaly-aware semantic segmentation to detect obstacles outside the predefined classes. This will allow for the identification of unknown obstacles as part of the segmentation output. To ensure that each identification is accompanied by a reliable measure of confidence, the framework will integrate uncertainty quantification through Conformal Prediction methods. By combining these components, the system will not only recognize potential obstacles but also provide formal statistical guarantees regarding the reliability of its predictions.

## Main Objectives

- **Anomaly-Aware Obstacle Segmentation**: integrate into a semantic segmentation model techniques to detect
obstacles that fall outside known classes.

- **Statistical Uncertainty Quantification**: btain semantic segmentation outputs and obstacle proposals guar-
antees on detection reliability.

- **Comprehensive Evaluation**: benchmark the system using both detection performance metrics and uncertainty
metrics.


## Project Links

- [Google Drive](https://drive.google.com/drive/folders/1wAAcfMKKd2QQCEiel5mYU2wyu8uhUWSR?usp=share_link)
- [Notion Page](https://www.notion.so/Presentation-1eea146c941d8017b40ec1013bf70646?pvs=4)


## Useful Links

### Datasets

- [CityScapes](https://www.kaggle.com/datasets/shuvoalok/cityscapes/data)
- [LostAndFound](https://wwwlehre.dhbw-stuttgart.de/~sgehrig/lostAndFoundDataset/index.html)
- [FishyScapes](https://fishyscapes.com)

To avoid overloading the Git repository, download the CityScapes dataset from the link above and ensure it has the following structure:

```
CV-HW/
└── cityscapes/
    ├── train/
    │   ├── img/     ← RGB images
    │   └── label/   ← segmentation masks
    └── val/
        ├── img/
        └── label/
```

In `globals.py`, if you import the dataset out of the src folder and name it `cityscapes`, ensure the path is set to:
```python
CITYSCAPES_PATH = "../cityscapes"
```


### Papers

- [Conformal Semantic Image Segmentation Post-hoc Quantification of Predictive Uncertainty](https://arxiv.org/pdf/2405.05145)
- [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/pdf/2107.07511)
- [Road Obstacle Detection based on Unknown Objectness Scores](https://arxiv.org/pdf/2403.18207)