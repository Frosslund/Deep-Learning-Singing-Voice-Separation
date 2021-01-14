# Singing Voice Separation in Musical Arrangements Using a Deep U-Net Convolutional Neural Network

Final project in the course DT2119 Speech and Speaker Recognition at KTH Royal Institute of Technology. Project focused on the task of separating the singing voice from the remaining arrangements in music. STFT- and Log Mel Spectrum-representations of musical data were fed into a modified U-Net CNN, and common metrics within the research field were utilized for evaluation. Data augmentations methods were analyzed and compared.

## Team Members

<ul>
    <li>
        <strong>Lukas Frösslund</strong> - <i style="text-decoration: none;">lukasfro@kth.se</i>
    </li>
    <li>
        <strong>Valdemar Gezelius</strong> - <i style="text-decoration: none;">vgez@kth.se</i>
    </li>  
</ul>

## Technologies

-   [Python 3](https://www.python.org/)
-   [TensorFlow 2](https://www.tensorflow.org/)
-   [Keras](https://keras.io/)
-   [SciPy](https://www.scipy.org/)
-   [Librosa](https://librosa.org/doc/latest/index.html)

## Project Details

The MUSDB18 dataset was used for this project. Preprocessing of the data included a transform to the frequency domain with an STFT, and conversion to the Log Mel Spectrum-representation. Mel Frequency Cepstral Coefficients (MFCCs) were avoided due to the belief that our network could handle the feature correlation well, and that an additional DCT decorrelation process therefore was not necessary.

Pitch Shifting, Time Stretching and Random Amplitude Scaling were all used and compared as augmentations methods. The binary cross-entropy loss was used as our loss function. As evaluation metrics, SAR and SDR was implemented.

More details available in the <a href="https://github.com/Frosslund/Deep-Learning-Singing-Voice-Separation/blob/main/Project_Report_DT2119.pdf">project report</a>

## Visual Data Representation

![data_rep](https://github.com/Frosslund/Deep-Learning-Singing-Voice-Separation/blob/main/images/data_rep.png?raw=true)
