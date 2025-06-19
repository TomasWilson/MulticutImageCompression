**NOTE: For privacy reasons, this is a stripped down version of the project, which is missing some further experiments (and the original commit history).**

## Multicut for Image Compression

This project is something I worked on in the winter semester 2024 at TU Dresden. 
The goal of the project was to create a novel image compression algorithm, that 
compresses an image using a so-called "multicut", which is effectively a clustering 
of the images pixels, and then encoding color information only once for each cluster, by encoding it's mean color. 

Allthough simple, this technique can compress images surprisingly well and, using the SSIM measure, 
it can actually be shown that this algorithm can get smaller file sizes at the same 
quality level as JPEG compression. (Note however that this method comes with a much higher computational effort.)

To optimize the compression, several methods for encoding the color and partition information are 
used and compared. They range from "naive" encodings, to rather sophisticated encoding schemes based on 
huffman- and arithmetic coding. 

### About the code

The main logic of the project has been written in C++, relying on `ZLIB`, `Boost`, `MLPACK` and `OpenMP` as dependencies. 
Additionally, to compare the algorithm against popular compression techniques, Python was used to 
summarize and display the results. (See the `intermediate-presentation` and `final-presentation` folders). 
For interoperability between C++ and Python, a subset of the C++ functionality was exported to Python, 
using `Boost.Python`. 

**Please note that large parts of this codebase are somewhat messy. This is because this project was, 
in many ways, very experimental and it is incredibly hard to find a software architecture that can cope 
with the rapid changes in program structure that are needed for such experimentation.**

### Usage
The code can be compiled using the provided `CMakeLists.txt`.  
