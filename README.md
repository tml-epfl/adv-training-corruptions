# On the effectiveness of adversarial training against common corruptions

**Klim Kireev\* (EPFL), Maksym Andriushchenko\* (EPFL), Nicolas Flammarion (EPFL)**

**Paper:** [https://arxiv.org/abs/2103.02325](https://arxiv.org/abs/2103.02325)

\*Equal contribution.



## Abstract
The literature on robustness towards common corruptions shows no consensus on whether adversarial training can improve 
the performance in this setting. First, we show that, when used with an appropriately selected perturbation radius, L<sub>p</sub> 
adversarial training can serve as a strong baseline against common corruptions. Then we explain why adversarial training 
performs better than data augmentation with simple Gaussian noise which has been observed to be a meaningful baseline on 
common corruptions. Related to this, we identify the *σ-overfitting phenomenon* when Gaussian augmentation overfits to a 
particular standard deviation used for training which has a significant detrimental effect on common corruption accuracy. 
We discuss how to alleviate this problem and then how to further enhance L<sub>p</sub> adversarial training by introducing an 
efficient relaxation of adversarial training with learned perceptual image patch similarity as the distance metric. 
Through experiments on CIFAR-10 and ImageNet-100, we show that our approach does not only improve the L<sub>p</sub> adversarial 
training baseline but also has cumulative gains with data augmentation methods such as AugMix, ANT, and SIN leading to 
state-of-the-art performance on common corruptions.



## About the paper
First of all, we observe that even L<sub>p</sub> adversarial training (e.g., for p in {2, inf}) can lead to significant improvements on common corruptions
and be competitive to other natural baselines:
<p align="center">
    <img src="images/linf_at_helps.png" width="350">
    <img src="images/l2_at_vs_natural_baselines.png" width="350">
</p>

Next, we discuss the *σ-overfitting phenomenon* when Gaussian augmentation overfits to a particular standard deviation used for training. 
This can be seen particularly clearly on ImageNet-100:
<p align="center"><img src="images/sigma_overfitting_imagenet.png" width="750"></p>
As we show in the experimental part, this leads to significantly suboptimal results on common corruptions that, however, 
can be improved by augmenting only 50% images per batch (as done, e.g., in Rusak et al, (2020)).
<br/>

Then we show how to improve adversarial training by using the LPIPS distance instead of the standard L<sub>p</sub> distances.
First, we discuss why LPIPS can be more suitable than L<sub>2</sub> norm on common corruptions. We observe that L<sub>2</sub> norm
does not always capture well the perturbation magnitude of common corruptions. For example, on several corruptions
(especially, on elastic transforms) L<sub>2</sub> norm is monotonically *decreasing* over corruption severity levels instead of increasing, 
while for LPIPS this happens less often:
<p align="center"><img src="images/l2_vs_lpips_distance_non_monotonic.png" width="475"></p>

This can be further quantified by computing the correlation between L<sub>2</sub>/LPIPS distances and error rates for some standard model:
<p align="center"><img src="images/lpips_is_better_correlated.png" width="800"></p>
If some corruptions make it harder for the network to classify examples correctly, this should be also reflected in a
larger perturbation magnitude.

Next, we present an efficient relaxation of LPIPS adversarial training which we call *Relaxed LPIPS Adversarial Training* (RLAT)
which can be efficiently solved using an FGSM-like algorithm:
<p align="center"><img src="images/lpips_relaxation.png" width="550"></p>

Finally, we present experiments where we show that RLAT achieves competitive results on common corruptions compared to 
the existing baselines. In particular, RLAT outperforms L<sub>p</sub> adversarial training and gives additional improvement
when combined with different data augmentation methods.
<p align="center"><img src="images/results_cifar10.png" width="750"></p>
<p align="center"><img src="images/results_imagenet100.png" width="750"></p>



## Code
The main dependencies are specified in `requirements.txt`.
 
To train new models, one can use the following commands:

- Standard training on CIFAR-10:
`python train.py --eps=0.0 --attack=none --epochs=150 --model_path='models/standard-cifar10.pt'`

- L<sub>2</sub> adversarial training with `eps=0.1` on CIFAR-10:
`python train.py --eps=0.1 --attack=pgd --distance=l2 --epochs=150 --model_path='models/l2-at-eps=0.1-cifar10.pt'`

- Relaxed LPIPS adversarial training (RLAT) with `eps=0.08` on CIFAR-10:
`python train.py --eps=0.08 --attack=rlat --distance=l2 --epochs=150 --model_path='models/rlat_eps=0.08-cifar10.pt'`

Training ImageNet models can be done similarly but you additionally need to have the ImageNet dataset in folder `<dataset_directory>/imagenet` 
where the default value of `<dataset_directory>` is `./datasets` (see `train.py`).



## Models
We provide all the models reported in Tables 3 and 4 in [this Google drive folder](https://drive.google.com/drive/folders/1drD6E3xX2ERjIuYoZjFleTZ7rr7WcYBq?usp=sharing). 

Example how to run evaluation of a model:
- `python eval_cifar10c.py --checkpoint=models_paper/cifar10/rlat-eps=0.08-cifar10.pt`
- `python eval_imagenet100c.py --checkpoint=models_paper/imagenet100/rlat-eps=0.02-imagenet100.pt`

Note that the CIFAR-10 Fast PAT model `fast-pat-eps=0.02-cifar10.pt` is the only exception and requires the 
[original code](https://github.com/cassidylaidlaw/perceptual-advex) to be properly restored.



## Full evaluation results
We additionally provide detailed evaluation results for each corruption type and each severity level for all CIFAR-10 
and ImageNet-100 models in folder [`full_evaluation_results`](https://github.com/tml-epfl/adv-training-corruptions/tree/main/full_evaluation_results).
