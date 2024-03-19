# [TAF](https://taf.mpi-inf.mpg.de/): An Implicit Neural Representation for the Image Stack: Depth, All in Focus, and High Dynamic Range [Siggrapha Aisa 2023]


### Installation
Use the following commands with Anaconda to create and activate your environment:
  - ```conda env create -f environment.yml```
  - ```conda activate TAF```


### Training
We follow the progressive training be strategy, strat from the low resolution and increase the resolution step by step.

```
 python train.py --save_path "./results" --base_path "./dataset/scene_1/"
```
please change the base_path to your own dataset path

### Testing
Continuing with pretrained stem:
```
python test.py 
```
please modify the wieghts path to your own for testing, here we include pre-trained weights in folder “weight” as an example.

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@article{wang2023implicit,
  title={An Implicit Neural Representation for the Image Stack: Depth, All in Focus, and High Dynamic Range},
  author={Wang, Chao and Serrano, Ana and Pan, Xingang and Wolski, Krzysztof and Chen, Bin and Myszkowski, Karol and Seidel, Hans-Peter and Theobalt, Christian and Leimk{\"u}hler, Thomas},
  journal={ACM Transactions on Graphics (TOG)},
  volume={42},
  number={6},
  pages={1--11},
  year={2023},
  publisher={ACM New York, NY, USA}
}
}</code></pre>
  </div>
</section>

### Acknowledge
This source code is derived from the (https://shnnam.github.io/research/nir/). We really appreciate the contributions of the authors to that repository.



