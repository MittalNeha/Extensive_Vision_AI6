# CAPSTONE Submissions

- [Problem Statement](#problem-statement)
- [DETR](#detr--detection-transform)
- [Encoder Decoder Architecture](#encoder-decoder-architecture)
- [Bipartite Loss](#bipartite-loss)
- [Object Queries](#object-queries)
- [Team Members](#team-members)


# Problem Statement

- We take the encoded image (dxH/32xW/32) and send it to Multi-Head Attention (**FROM WHERE DO WE TAKE THIS ENCODED IMAGE?**)
- We also send dxN Box embeddings to the Multi-Head Attention
- We do something here to generate NxMxH/32xW/32 maps. (**WHAT DO WE DO HERE?**)
- Then we concatenate these maps with Res5 Block (**WHERE IS THIS COMING FROM?**)
- Then we perform the above steps (**EXPLAIN THESE STEPS**)
- And then we are finally left with the panoptic segmentation



<p align="center">
  <img src="images/arch.png" alt="drawing">
</p>

Some details are not clear about this image which has been taken from the paper [End-to-End Object Detection with Transformers](https://arxiv.org/pdf/2005.12872.pdf). In this section of the paper, the author talks about using DETR for panoptic segmentation. It says that DETR can be extended for panoptic segmentation by adding a mask head to the decoder output. Quoting the paper on the same, *Panoptic segmentation unifies the typically distinct tasks of semantic segmentation (assign a class label to each pixel) and instance segmentation (detect and segment each object instance).*

If we just look at the images in this architecture, we see that the input image is converted to attention maps, which are then converted into binary masks, which are then merged to create the panoptic segmentation. 

To begin with, the input is 3xHxW, which is 3 channels. DETR has the Resnet50 backbone, that converts the input image to 2048 x H/32 x W/32. This is then projected to match the hidden dimension of the transformer network which is d and is 256 by default. This goes through the transformer encoder and creates a d x H/32 x W/32 Encoded Image.

The decoder output which is dxN class predictions is passed through a Multi head attention with M heads to get N x M x H/32 x W/32 Attention maps. To elaborate this, an attention function is applied on the encoded images, which is nothing but the encoder output. This means that the encoded images form the query and key vectors for the multi head attention. The decoder output, which is the N output embeddings for the predictions, is fed as queries to this block.

<p align="center">
  <img src="images/attention.png" alt="drawing">
</p>

The output from this is N x M x H/32 x W/32. Here N is the class predictions. So these are the N attention maps, where each class has M classes and is of dimension H/32 x W/32.



Next, the resolution of these attention maps need to be increased so that it later can be masked on the image of original resolution. A FPN (Feature Pyramid Network) style CNN is used for the same which gets residuals from the backbone ResNet network, which was used to downscale the dimensions to begin with.

The basic architecture of FPN looks like this:

<p align="center">
  <img src="images/FPN.png" alt="drawing">
</p>

This has 2 pathways, one Bottom-Up which is basically the ResNet50 backbone used in DETR and the other is the Top Down Pathway, used to create the Masks Logits.

<More explanation along with a diagram to follow>

Finally, a pixel-wise argmax is taken of each of the N predicted H/4 x W/4 mask logits, which gives the final 
