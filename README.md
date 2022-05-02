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
  - [Contributor](#Contributor)
  - [License](#License)

## Background
> The standard methods of image compression come in several varieties. The current most popular methods relies on eliminating high frequency components of the signal by storing only the low frequency Fourier coefficients. Other methods use a "building block" approach, breaking up images into a small number of canonical pieces and storing only a reference to which piece goes where.  <br />
- Fractals scheme is promoted by M. Barnsley

## Demo

### SVD
- Lena's image (512 * 512)
![](https://github.com/MRSA-J/Fractal_Image_Compression_40/blob/main/data/lena.jpg)                     <br />
* above: original                                                                                       <br />
![](https://github.com/MRSA-J/Fractal_Image_Compression_40/blob/main/data/lena_svd_generated_200.jpg)   <br />
* above: compressed                                                                                     <br />
### Fractal Image Compression

![](https://pandao.github.io/editor.md/images/logos/editormd-logo-180x180.png)    <br />


> Todo: modify

## Feature & Todo List
- [x] SVD
- [x] Fractal Image Compression
    - [ ] Support more transformations
- [x] Autoencoder (VAE)
- [ ] JPEG
- [ ] Modify readme
                
                    
## Comparison
| Method                     | Compression Ratio              | Settings & Note                |
| -------------------------- | ------------------------------ | ------------------------------ |
| SVD                        | 0.782012939453125              | singular value limit - 200     |
| Fractal Image Compression  | Cannot compute                 |                                |
| JPEG                       | Not implement yet              |                                |


## Function
| Function name | Description                    |
| ------------- | ------------------------------ |
| `help()`      | Display the help window.       |
| `destroy()`   | **Destroy your computer!**     |



## Contributor
- Code：   [@ChenWei](https://github.com/MRSA-J)  [@LinYu](https://github.com/linyu26)


## License
[MIT](LICENSE)
