# Ferienakademie2021_xray_ct
Practical Project: X-ray CT 

## Project summary
Implement from scratch a simulator of X-ray CT as well as classical tomographhic reconstruction methods to solve the ill-posed
inverse problem. Test out different deep learning approaches for tomographic reconstruction to see how they work compared to classical methods.

## Project goals:
* to understand the inner workings of CT better
* to get some experience with ill-posed inverse problem
* to see hoe deep learning can be applied in X-ray CT

## Project steps:
1. Implement a simulator of X-ray CT for 2D from scratch.
	* how to represent the geometry of a 2D X-ray CT setup?
	* how to represent the object to be imaged?
  	* how to trace Xrays thorugh the object to be imaged, i.e how to compute line integrals?
  	* implement a forward projection operation to simulate a CT aquisition (sinogram)
  	* https://gitlab.lrz.de/IP/elsa
  
2. Implement an analytic and an interative reconstruction method from scratch.
  	* analytic reconstruction: do the filtered back-projection (FBP) as in the seminar talks
		* you will need a back-projection operator (counterpart of the forward-projection)
	* iterative reconstruction: choose some iterative solver (e.g. CG) as in the seminar talks
	

3. ..  missing

4.

5.
