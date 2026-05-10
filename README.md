# SwinJSCC

This repository refactor the SwinJSCC code by pytorch>2.9 to fit in Sionna channel settings. Besides, I also supports training options with AMP and DDP. The metrics display the adjusted CBR under the situation that the mask needs to be transmitted through channel as well. I use AMC standard LDPC coding for the mask's cbr calculation. You can remove it under test.py if you don't like it.
