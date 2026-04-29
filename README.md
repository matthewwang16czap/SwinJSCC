# SwinJSCC

This repository refactor the SwinJSCC code by pytorch>2.9 to fit in Sionna channel settings. Besides, I also supports training options with AMP and DDP. You may observe the training result psnrs are slightly lower than the original paper results--This is completely normal especially looking at the other metrics are nearly identical. Anyways, I create this repository purely for benching SwinJSCC method on a widely accepted communication systems simulation platform instead of self coded simulation.
