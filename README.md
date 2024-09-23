# Sound Analysis Application 

- [Keyword Spotting (KWS)](KWS/)

# Introduction to Sound Analysis 

![KWS Archh](doc/KWS-architecture.svg)

Spectral features of sound has very unique signatures that can be leveraged in many applications including Keyword Spotting (KWS), preventive maintenance for heavy machinery and heart disease diagnosis.

Mel frequency cepstral coefficient (MFCC) is one of the most popular method for classifying sound signals. A typical _digital_ implementation of a sound classification is shown in the above figure used for kyword spotting (KWS) for extracting speech features such as a simple word "Alexa". In a typical all-digital immplementation, a digital microphone is used to read real-time data using an $I^2S$ serial interface. The serial data is converted to parallel bytes. 

The first operation done on the input data is a pre-emphasis filter (high-pass or HPF) to remove the DC content from the signal. The pre-emphasis filter is typicall implemented as the following difference equation: 

$y(n) = x(n) - \alpha x(n-1)$

where $\alpha$ ranges from 0.9-1. To keep the hardware minimal, the fraction is implemented as a shift+add. For eg. $y(n) = x(n) - ( x(n) - x(n)/32 )$ which is a shift and add operation to realize $\alpha = 31/32 = 0.96875$

After the pre-emphasis filter, the data is multiplied with a _window (hamming, hanning, etc.))_ to avoid spectral leakage from FFT operation. After the windowing, fast-fourier transform (FFT) is applied to the signal to find the frequency content of the signal. Then the linear frequency scale if converted to _Mel Log scale_ ( $Mel(f) = 2595 \cdot log_{10}(1 + f/700)$ ) to mimic the human ear perception. Then the _log_ of Mel frequency power is calculated and the DCT operation is done to generate the MFCC co-efficients. Finally, the MFCC co-efficients are used to classify the voice signal using a classifier such as Convolution Neural Netwrk (CNN), etc.

Typically the MFCC and the classifier are implemented on the _Edge Node_ using a microcontroller. But with more and more computing moving to the edge, we are now at a point where some of the trivial or not-so-trivial computing to move to the sensor itself. In this work we propose to move some of the front-end signal processing (HPF, wondowing and FFT) to the microphone itself which may call as _Bleeding-Edge Computing_. 


