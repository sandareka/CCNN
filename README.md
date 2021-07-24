# Comprehensible Convolutional Neural Network via Guided Concept Learning

Official implementation of Comprehensible Convolutional Neural Network via Guided Concept Learning
accepted to IJCNN 2021

## Abstract
Learning concepts that are consistent with human perception is important for Deep Neural Networks to win end-user trust. Post-hoc interpretation methods lack transparency in the feature 
representations learned by the models. This work proposes a guided learning approach with an additional concept layer in a CNN-based architecture to learn the associations between visual features and word phrases. We design an objective function that optimizes both prediction accuracy and semantics of the learned feature representations. Experiment results demonstrate that the proposed model can learn concepts that are consistent with human perception and their corresponding contributions to the model decision without compromising accuracy. Further, these learned concepts are transferable to new classes of objects that have similar concepts.

Presentation Video : <https://www.youtube.com/watch?v=vK4vti_pUMg>

## Overview of Comprehensible CNN
![ccnnOverview](https://github.com/sandareka/CCNN/blob/main/images/CCNN_Overview.jpg?raw=true )

## Citation
```
@article{wickramanayake2101learning,
  title={Comprehensible Convolutional Neural Networks via Guided Concept Learning},
  author={Wickramanayake, Sandareka and Hsu, Wynne and Lee, Mong Li},
  journal={arXiv preprint arXiv:2101.03919},
  year={2021}
}
```