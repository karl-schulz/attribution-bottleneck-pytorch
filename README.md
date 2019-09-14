# Info-Attribution
Developing attribution methods for DNNs by measuring/restraining the information flow through the network.

Evaluation method details: see ![evaluate/](attribution_bottleneck/evaluate/)

# Fast Installation
* git clone https://this-url.com
* cd attribution-bottleneck-pytorch
* pip install --user -e ./
* 

# Overview
1. Measuring layer activation probabilities for a input by comparison with a prior distribution, then converting probabilities to information content

    1.1. Prior is a histogram over previous activations in a layer
    
    1.2. Prior is a histogram over previous activations in a layer, seperate for each feature map
    
    1.3. Prior is a approcimate dirac at zero + right-sided half gaussian for > 0
  
2. Supress the flow of information with bottleneck layers

    2.1. Train injected bottleneck layer to discard unimportant informations/activations for a specific sample
  
    2.2. Train multiple bottlenecks at different depths in the network
  
3. Training a bottleneck on full dataset at once by using higher level information, which is optained by passing the sample through the network once before

## #1.1 Basic Approach with channel-wise histograms
See 1.2 but not channel-wise, only layer-wise.
## #1.2 Basic Approach with channel-wise histograms
At K different layers of different depth in the network, first empirically aproximate the distribution of the latents z for each feature map dimension by a histogram.
### Data Collection Phase
* Pass N input samples x through the network and collect the activations for each feature map.
* Build a histogram of the activations for each layer and each feature map, rescale them to have sum 1. 
### Attribution
* Pass a single sample through the network to analyse it. 
For each fetaure map in each layer:
* The probability of an activation z is approximated to be the bin probability. If it is outside the histogram range, it is approximated to be in a bin with one element.
* Map all activations to "information maps" - k:layer, c:feature map, p: bin probability in the histogram  
![equation](https://latex.codecogs.com/gif.latex?%24I%28z%5E%7B%28k%29%7D_c%29%20%3D%20-%5Clog%28p%28z_c%5E%7B%28k%29%7D%29%29%29%24)
* Scale these maps to cover the interval [0,1]

Now we have a information map for each feature map in each observed layer. To obtain a 2D heatmap, we sum up the information maps over the feature maps in each layer, rescale them to have corresponsing size and multiply them element-wise.  
![equation](https://latex.codecogs.com/gif.latex?%24S%28x%29%20%3D%20%5Csum_%7BC_1%7D%20I%28z_c%5E%7B%281%29%7D%29%20%5Codot%20...%20%5Codot%20%5Csum_%7BC_K%7D%20I%28z_c%5E%7B%28K%29%7D%29)

### Example
Top row: Information maps, meaned over the feature dimensions.  
Bottom row: multiplied feature maps over the layers.  
Left top: Original input  
Left bottom: Generated 2D heatmap  

![example](demo/channel-histograms/dog.png)

Some other examples:  
![example](demo/channel-histograms/imagenet_samples.png)

## #1.3 Basic Approach with fitted density function
Like #1.1, but no histograms to aproximate the prior distribution of activations. Instead, the distribution of x is approximated by 
* the probability of being zero p_0
* a half-gaussian distribution if x > 0 with peak/mean at 0

### Results
Similar results like the histogram approach, but tuning of the algorithm needed: Outliers cause extreme information peaks.

## #2.1 Limiting information flow with a bottleneck layer
Instead of just measuring the activations and compare them to a prior, measure which informations can be "removed" while maintaining the classification score.

A bottleneck layer is a layer similar to a VAE bottleneck, where the ouput z is sampled from a gaussian distribution with mean a*x and variance e², where x activations of the original layer:

z ~ N(ax, e²)

The training objective is extended to minimize the KL divergence between N(a, e²) and N(0,1), thus encouraging a damped/removed x and a noise with variance 1: 

L = NLL + KL( N(a, e²) | N(0,1) )

In order to be independent of the scale of activations x, x is preprocessed to have mean 0 and variance 1. After the bottleneck layer, the output z is rescaled to restore the original properties of x:

x = (x_in - mean_x_in) / std_x_in
z ~ N(ax, e²)
z_out = z * std_x_in + mean_x_in

### Attribution algorithm
* Take a pretrained network which we want to analyse
* Inject a bottleneck layer and initialize a = 1 and log(e) << 0 to have initially neutral behavior
* Pass through the sample we want to analyse and verify that original performance of the network is unharmed
* Train *only the bottleneck* network on the objective argmin L = NLL + KL( N(a, e²) | N(0,1) )
* Stop training on convergence or a fixed number of iterations
* The KL divergence term can now be used as an indicator where in the network informations was still passing through the bottleneck. Only "important" neurons should not have a=0 and e=1
* Reduce the feature dimension of the KL tensor by taking the average to obtain a 2D heatmap

### Qualitative Evaluation
Resulting heatmaps seems to capture well the inportant regions in the input. 
![evaluation](demo/bottleneck-single/pigs.png)

### Quantitative Evaluation
Even without full hyperparameter tuning, the method performs state-of-the-art. However, it is considerably slower than other methods (7s per sample on our machine).
![evaluation](demo/bottleneck-single/quant_vgg_single_1000.png)

## #2.2 Multiple Bottleneck Layers

TODO

## #3 Recurrent Bottleneck

TODO
