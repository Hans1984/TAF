# [Cinematic](https://cinegs.mpi-inf.mpg.de/): Cinematic Gaussians:  Real-Time HDR Radiance Fields with Depth of Field [Pacific Graphics 2024]


### Installation
Use the following commands with Anaconda to create and activate your environment:
  - ```conda env create -f environment.yaml```
  - ```conda activate cinegs```
Note: Our installation method is the same as that of [3DGS](https://github.com/graphdeco-inria/gaussian-splatting/tree/main). If you encounter installation issues, please refer to the issues section of that.

### Training
For real dataset
```
python train.py -s data_path -m output_path --fd_path ./exif_info_txt/.. --ap_path ./exif_info_txt/.. --exp_path  ./exif_info_txt/.. --length_focal .. --blur
```

For rendering dataset
```
python traing.py -s data_path -m output_path --length_focal .. --start_checkpoint pretrained_model_apth/chkpnt7000.pth --blur
```

please change the data_path to your own dataset path

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



