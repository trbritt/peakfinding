# peakfinding
This repository generates points in a reciprocal lattice and matches them to peaks autodetected in an image.
Images are automatically gamma corrrected to assist in peak identification. 'Black and White' 
images, or ndarray ([M[, N[, ...P]][, 1]), are automatically gamma corrected but require a different function which 
doesn't collapse multiple channels. 

![Fig 1: RGB Peak Detection and Lattice Indexing](test_RGB_fitted.jpg)