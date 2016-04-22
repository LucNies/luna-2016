# luna-2016

Notes van meeting met Mohsen:
Fully convolutional methods for image segmentation
Replace fully connected layer with convolution layer. How? Kernel = height*width*depth
Then upsample for output
http://www.cs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

*Unet in caffe*
(Normally train on patches, test on image for segmentation, Unet does it differently)
http://arxiv.org/pdf/1505.04597.pdf

*Random tips*
No padding! 

*Patch size:* 
+large is more contextual information. 
-More computation (larger convolution).
-Too coarse, so it has a hard time with borders.

*Pooling?*
Don't. We want big output (we don't use fully-connected anyway)
Even when patching, we don't want max pooling since it will confuse 'lung pixels' from pixels other than the center on (the only relevant pixel since we want to classify it) and misclassify more often.

*Sampling*
Balanced and sample from border. Throw away easy samples from the outside of the image and the center of the segment.

**Eigen ideeen**
Harmen) Deconvolution for segmentation (al gedaan maar vind Mohsen vast cool)
http://arxiv.org/abs/1505.04366
>put whole image in
>similar to unet

Luc) Multi-scale approach
Combine networks with different scales.

Division of tasks:
Tom & Luc: Fully convolutional
Harmen: Unet
Inez & Steven: Classical methods
