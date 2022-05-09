# Math_40_Final_Project


## Fractal Image Compression Introduction
- Paper: Yuval Fisher, [Fractal image Compression](https://moodle2.brandeis.edu/pluginfile.php/2743626/mod_folder/content/0/1992_Fisher.pdf?attredirects=0), 1992.
- Fractal compression is a lossy compression method for digical images based on fractals. The method is best suited for textuals & natural images, relying on the fact that parts of an image often resemble other parts of the same image. Fractal algorithm convert these part into mathematical data called "fractal code" which are used to recreate encoded image

## Table of Contents
  - [Fractal Image Compression](#Fractal-Image-Compression-Introduction)
  - [Background](#Background)
  - [Demo](#Demo)
    - [SVD](#SVD)
    - [Fractal Image Compression](#Fractal-Image-Compression)
    - [Autoencoder](#Autoencoder)
  - [Contributor](#Contributor)
  - [License](#License)

## Background
> The standard methods of image compression come in several varieties. The current most popular methods relies on eliminating high frequency components of the signal by storing only the low frequency Fourier coefficients. Other methods use a "building block" approach, breaking up images into a small number of canonical pieces and storing only a reference to which piece goes where.  <br />
- Fractals scheme is promoted by M. Barnsley

## Demo

### SVD
- Lena's image (512 * 512)                                                                                       <br />
![](https://github.com/MRSA-J/Fractal_Image_Compression_40/blob/main/data/lena.jpg)                              <br />
* Above: Original                                                                                                <br />
![](https://github.com/MRSA-J/Fractal_Image_Compression_40/blob/main/data/lena_svd/lena_svd_generated_200.jpg)   <br />
* Above: Compressed with # singular value limit 200                                                              <br />

| Sigular Value limit    | Compression Ratio      |  Generated Image                                                                                              |
| ---------------------- | -----------------------| ------------------------------------------------------------------------------------------------------------- |
| 200                    | 0.782012939453125      | [Lena 200](https://github.com/MRSA-J/Fractal_Image_Compression_40/blob/main/data/lena_svd/lena_svd_generated_200.jpg)  |
| 180                    | 0.7038116455078125     | [Lena 180](https://github.com/MRSA-J/Fractal_Image_Compression_40/blob/main/data/lena_svd/lena_svd_generated_180.jpg)  |
| 160                    | 0.6256103515625        | [Lena 160](https://github.com/MRSA-J/Fractal_Image_Compression_40/blob/main/data/lena_svd/lena_svd_generated_160.jpg)  |
| 140                    | 0.5474090576171875     | [Lena 140](https://github.com/MRSA-J/Fractal_Image_Compression_40/blob/main/data/lena_svd/lena_svd_generated_140.jpg)  |
| 120                    | 0.469207763671875      | [Lena 120](https://github.com/MRSA-J/Fractal_Image_Compression_40/blob/main/data/lena_svd/lena_svd_generated_120.jpg)  |
| 100                    | 0.3910064697265625     | [Lena 100](https://github.com/MRSA-J/Fractal_Image_Compression_40/blob/main/data/lena_svd/lena_svd_generated_100.jpg)  |
|  80                    | 0.31280517578125       | [Lena  80](https://github.com/MRSA-J/Fractal_Image_Compression_40/blob/main/data/lena_svd/lena_svd_generated_80.jpg)  |
|  60                    | 0.2346038818359375     | [Lena  60](https://github.com/MRSA-J/Fractal_Image_Compression_40/blob/main/data/lena_svd/lena_svd_generated_60.jpg)  |
|  40                    | 0.156402587890625      | [Lena  40](https://github.com/MRSA-J/Fractal_Image_Compression_40/blob/main/data/lena_svd/lena_svd_generated_40.jpg)  |
|  20                    | 0.0782012939453125     | [Lena  20](https://github.com/MRSA-J/Fractal_Image_Compression_40/blob/main/data/lena_svd/lena_svd_generated_20.jpg)  |


As we could see from above, human eyes cannot easily see the difference between the original and the generated image. Moreover, SVD doesn't require large dataset to train. It can be directly used without training.    <br />

> Also, we could see that the fewer singular value we use, the worse our performance.


### Fractal Image Compression
- Lena's image (512 * 512)                                                                                                <br /> 
![](https://github.com/MRSA-J/Fractal_Image_Compression_40/blob/main/data/lena_fractal/lena_fractal_generated.jpg)        <br />
- Monkeys's image (256 * 256)                                                                                             <br />
![](https://github.com/MRSA-J/Fractal_Image_Compression_40/blob/main/data/monkey_fractal/monkey_fractal_generated.jpg)    <br />

### Autoencoder
- Note: Autoencoder can only be trained on a dataset which contains similar objects. This is because autoencoders learn how to compress the data based on attributes (i.e. correlations between the input feature vector) discovered from data during training. 
- Trained on MNIST in our code


## Feature & Todo List
- [x] SVD
- [x] Fractal Image Compression
    - [ ] Support more transformations
- [x] Autoencoder (VAE)
- [ ] JPEG
- [x] Modify readme
                
                    
## Comparison
| Method                     | Compression Ratio              | Settings & Note                     |
| -------------------------- | ------------------------------ | ----------------------------------- |
| SVD                        | 0.782012939453125              | singular value limit - 200          |
| Fractal Image Compression  | Cannot compute                 |                                     |
| JPEG                       | Not implement yet              | Does not have time to implement : ( |

The 2 typical methods among all data/image compression methods are:   </br>
- 1. Using the NN network
- 2. Using some maths methods (i.e. fractal image compression, SVD, PCA, etc). 

</br>
Difference:   </br>
- Dataset     
  - NN network: requires a huge dataset, labeled or unlabeled. Takes long to train.
  - SVD, fractal image compression: Only requires a single image. 
- 'Decoding'/reconstruction algorithm  
  - NN network: Compute very quickly using the weight of each layer.
  - SVD, fractal image compression: Use an iterative reconstruction, which is normally computationally expensive.


## Function
| Function name                      | Description                                |
| ---------------------------------- | ------------------------------------------ |
| `python svd.py`                    | SVD methods                                |
| `python autoencoder.py`            | Auto encoder implementation                |
| `python fractal_compress.py`       | Fractal Image Compression implementation   |

- Make sure you have the required package (pytorch, skimage, scipy, etc) downloaded before you run the project

## Contributor
- Code：   [@ChenWei](https://github.com/MRSA-J)  [@LinYu](https://github.com/linyu26)


## License
[MIT](LICENSE)
