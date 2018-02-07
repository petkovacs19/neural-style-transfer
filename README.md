# Neural Style Transfer

This is a tensorflow implementation of neural style transfer. Using an intersting approach shown by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge, you can use convolutional nerual networks to transfer styles of images to another image. Using convolutional layers we can generate a new image with the content of one and the style of another one.

### Usage 

```
python neural_style_transfer.py -content content.jpg -style style.jpg -iterations 1000 -lr 2.0 -vgg_path imagenet-vgg-verydeep-19.mat
```

Using the Great Church of Debrecen as content image and applying the style of The Starry Night by Vincent van Gogh, we can generate a new image of Debrecen.
<div align="center">
	<img src="https://github.com/petkovacs19/neural-style-transfer/blob/master/images/debrecen.jpg"/>
	<img src="https://github.com/petkovacs19/neural-style-transfer/blob/master/images/vincent.jpg"/>
</div>

<div align="center">
	<img src="https://github.com/petkovacs19/neural-style-transfer/blob/master/images/debrecen-vincent.png"/>
</div>



Similarly we can have a look at London in a cubist style

<div align="center">
	<img src="https://github.com/petkovacs19/neural-style-transfer/blob/master/images/london.jpg"/>
	<img src="https://github.com/petkovacs19/neural-style-transfer/blob/master/images/picasso.jpg"/>
</div>

<div align="center">
	<img src="https://github.com/petkovacs19/neural-style-transfer/blob/master/images/london-picasso.png"/>
</div>


### Setup

1. You need to download the pretained [VGG-19](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat) neural network.

2. Install the dependencies and start the notebook

    ```sh
    $ jupyter notebook
    ```

3. Pass the path of the pretained network to get_vgg_graph function



### References
* Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).
* Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
* Vincent Dumoulin, Francesco Visin - A guide to convolution arithmetic for deep learning  https://github.com/vdumoulin/conv_arithmetic
* TensorFlow Implementation of "A Neural Algorithm of Artistic Style". https://github.com/log0/neural-style-painting
* Zeiler, Matthew D., and Rob Fergus. "Visualizing and understanding convolutional networks." European conference on computer vision. Springer, Cham, 2014.

License
----

MIT
