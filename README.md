# starGANv1-Pytorch
<b>paper:</b>https://arxiv.org/pdf/1711.09020.pdf<br>
<b>official PyTorch implementation:</b>https://github.com/yunjey/stargan<br>
StarGANv1 is a scalable image-to-image translation model among multiple domains using a single generator and a discriminator.Although the StarGANv2 is proposed in another paper,I think the function of StarGANv1 is good.StarGANv1 of paper can be applied by to multiple datasets.The official PyTorch implementation is difficult to understand for beginners.As a consequence,I wrote a simpler version of the code and I want to make my version of StarGANv1 public for those who are looking for an easier implementation of the paper.
## Requirements
* The code has been written in Python (3.6.9) and PyTorch (1.4)
## How to run
* To run training
```Python
python main.py --mode train
```
* To run testing
```Python
python main.py --mode test
```
