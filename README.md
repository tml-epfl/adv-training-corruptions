# On the effectiveness of adversarial training against common corruptions

**Klim Kireev\* (EPFL), Maksym Andriushchenko\* (EPFL), Nicolas Flammarion (EPFL)**

**Paper:** [https://arxiv.org/abs/2103.02325](https://arxiv.org/abs/2103.02325)

\* Equal contribution.





## Abstract
The literature on robustness towards common corruptions shows no consensus on whether adversarial training can improve 
the performance in this setting. First, we show that, when used with an appropriately selected perturbation radius, ℓp 
adversarial training can serve as a strong baseline against common corruptions. Then we explain why adversarial training 
performs better than data augmentation with simple Gaussian noise which has been observed to be a meaningful baseline on 
common corruptions. Related to this, we identify the σ-overfitting phenomenon when Gaussian augmentation overfits to a 
particular standard deviation used for training which has a significant detrimental effect on common corruption accuracy. 
We discuss how to alleviate this problem and then how to further enhance ℓp adversarial training by introducing an 
efficient relaxation of adversarial training with learned perceptual image patch similarity as the distance metric. 
Through experiments on CIFAR-10 and ImageNet-100, we show that our approach does not only improve the ℓp adversarial 
training baseline but also has cumulative gains with data augmentation methods such as AugMix, ANT, and SIN leading to 
state-of-the-art performance on common corruptions.



## About the paper
Coming soon.



## Code
The main requirements are specified in `requirements.txt`

Train clean model:
`python train.py --eps 0.0 --attack none --epochs 150 --data_dir ../datasets/ --model_path models/clean.pt`

Train L2 AT model:
`python train.py --eps 25.5 --attack rlat --distance l2 --epochs 150 --data_dir ../datasets/ --model_path models/l2_0.1.pt`

Train RLAT model:
`python train.py --eps 25.5 --attack rlat --distance l2 --epochs 150 --data_dir ../datasets/ --model_path models/rlat_0.1.pt`

Epsilon should be specified multiplied by 255.0



### Models
Coming soon.
