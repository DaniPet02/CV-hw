# CV-hw
Uncertainty-Aware Road Obstacle Identification

## Abstract

Reliable road obstacle identification is a critical requirement for the safe operation of autonomous driving systems. Traditional object detection methods often struggle to recognize unexpected or unknown obstacles, as they are typically limited to predefined categories. The ability to detect obstacles beyond known classes, particularly in dynamic and complex environments, is essential for the safety of autonomous vehicles. Recent advancements in semantic segmentation, anomaly detection, and uncertainty quantification offer new avenues to improve detection accuracy and reliability, enabling systems to recognize both known and unknown road obstacles. Such uncertainty-aware methods provide formal statistical guarantees on the reliability of predictions, a crucial aspect for ensuring safe and robust decision-making in real-world driving conditions.


## The Task

The aim of this project is to develop a general, model-agnostic framework for road obstacle identification, starting from the outputs of any semantic segmentation network. The system will focus on anomaly-aware semantic segmentation to detect obstacles outside the predefined classes. This will allow for the identification of unknown obstacles as part of the segmentation output. To ensure that each identification is accompanied by a reliable measure of confidence, the framework will integrate uncertainty quantification through Conformal Prediction methods. By combining these components, the system will not only recognize potential obstacles but also provide formal statistical guarantees regarding the reliability of its predictions.

## Main Objectives

- **Anomaly-Aware Obstacle Segmentation**: integrate into a semantic segmentation model techniques to detect
obstacles that fall outside known classes.

- **Statistical Uncertainty Quantification**: obtain semantic segmentation outputs and obstacle proposals guar-
antees on detection reliability.

- **Comprehensive Evaluation**: benchmark the system using both detection performance metrics and uncertainty
metrics.

## Useful Links

### Datasets

- [CityScapes Website Download](https://www.cityscapes-dataset.com/downloads/)
- [LostAndFound Website Download](https://wwwlehre.dhbw-stuttgart.de/~sgehrig/lostAndFoundDataset/index.html)
- [RoadAnomaly Website Download](https://www.epfl.ch/labs/cvlab/data/road-anomaly/)
- [CS GitHub](https://github.com/mcordts/cityscapesScripts/tree/master)

THe main script to execute the pipeline is in the `main_notebook.ipynb`, while all utils functions are in the `utils.py` file, to avoid overloading the notebook and keep everything cleaner.
Furthermore, to avoid overloading the Git repository, download all useful datasets from their official websites. Specifically, after having registered on the cityscapes website, download the CityScapes datasets `leftImg8bit_trainvaltest.zip (11GB)` for the RGB images and `gtFine_trainvaltest.zip (241MB)` for the masks, from the link `CityScapes Website Download`, add both folders to a new one called `cityscapes`, and ensure it has the following structure:

```
CV-HW/
└── cityscapes/
    ├── img/     ← RGB images
    │   ├── test/     
    │   ├── train/   
    │   └── val/
    └── mask/   ← segmentation masks
        ├── test/   ← empty
        ├── train/
        └── val/
```

Do the same with the LostAndFound dataset: download `leftImg8bit.zip (6GB)` for the RGB images and `gtCoarse.zip (37MB)` for the masks, from the link `LostAndFound Website Download`, add both folders to a new one called `lostandfound`, and ensure it has the following structure:

```
CV-HW/
└── lostandfound/
    ├── img/     ← RGB images
    │   ├── test/     
    │   ├── train/   
    └── mask/   ← segmentation masks
        ├── test/
        ├── train/
```

Same again with the RoadAnomaly dataset: download `RoadAnomaly.zip[WebP,13MiB]` for the RGB images from the linked website `RoadAnomaly Website Download`.

Then, by running the functions `fix_cityscapes`, `fix_lostandfound` and `fix_roadanomaly` you will have all three datasets fixed in the right way, without subfolders, and with nicer files names.
In the `globals` section, if you import the datasets out of the src folder, ensure to set the right relative path:
```python
relative_path = "../"
```


### References

- [Conformal Semantic Image Segmentation Post-hoc Quantification of Predictive Uncertainty](https://arxiv.org/pdf/2405.05145)
- [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/pdf/2107.07511)
- [Road Obstacle Detection based on Unknown Objectness Scores](https://arxiv.org/pdf/2403.18207)
- [Rethinking Atrous Convolution for Semantic Image Segmentation (DeepLabV3+)](https://arxiv.org/pdf/1706.05587v3)
- [Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)