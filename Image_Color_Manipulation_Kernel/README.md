# Image Color Manipulation (greyscale)
This CUDA kernel performs grayscale conversion on a jpg image, using weighted averages of RGB channels to determine pixel intensity. It loads the image, transfers data to the GPU, executes the CUDA kernel for grayscale conversion, and saves the resulting image back to disk with "_grey" appended to the filename.

![Image](https://github.com/CorrelateVisuals/Nvidea_CUDA/blob/main/Image_Color_Manipulation_Kernel/galaxy.jpg?raw=true)
Original © GRANTECAN S.A.
```
Object Name: 	Stephan's Quintet
Telescope:	Grantecan / Nasmyth-B
Instrument:	OSIRIS
Filter:	        G (481nm), R (641nm), I (770 nm), Z (970nm), OS657 (657nm, FHWM 35nm)
Color:	        Blue (G), Green (Blue+Red), Red (R+I+Z), Yellow (OS657)
Exposure:	4 x 30 secs (G, R, I, Z), 4 x 90 secs (OS657)
Field of View:	Approx. 5' x 6'
Orientation:	North is up, East is left
Position:	RA(J2000.0) = 22h35m57s
Dec(J2000.0) =  33°57'36"
Image processing: Daniel López/IAC
```
![Image](https://github.com/CorrelateVisuals/Nvidea_CUDA/blob/main/Image_Color_Manipulation_Kernel/galaxy_grey.jpg?raw=true)
Greyscale © GRANTECAN S.A.

