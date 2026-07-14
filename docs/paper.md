# Image Representation of Acoustic Features for the Automatic Recognition of Underwater Noise Targets
**2012 Third Global Congress on Intelligent Systems**
Zeng Xiangyang, He Jiaruo, Ma Lixiang
College of Marine Engineering, PB 58, Northwestern Polytechnical University Xi'an, 70072, PR China

## Abstract
Feature extraction is one of the most important technologies for underwater targets recognition. In the past few decades, a number of methods for feature extraction have been developed, and under certain conditions they can achieve high recognition rate. However, for complex environments, it is still difficult to improve the robustness of the recognition system, and new robust feature extraction methods are expectant. This paper presents a novel method of feature extraction based on the spectrogram of acoustic signals. The image moment features and image texture features are extracted and the algorithms of LDA, PCA and their combinations are used to select the effective features respectively. The experimental results show that, these selected image features can achieve high recognition rate.

**Keywords**: Underwater noise targets; feature extraction; image representation; feature selection

---

## I. INTRODUCTION
Recognition of underwater noise targets is significant not only in military areas, but also in economic fields including the detection of petroleum, fish school and other marine resources. Feature extraction and automatic classification for the signals collected by sonar are two important procedures in underwater noise targets recognition. At present, a number of techniques have been brought forward including the extraction of wave features, spectral features, time-frequency features[1] and auditory features[2]. They can achieve high recognition rate under certain conditions. However, for complex environments, it is still difficult to improve the robustness of the recognition system. Therefore, it is still an open issue to find new feature extraction approaches.

An acoustic signal contains the information of a physical variable with the change of time, and is usually one-dimensional function, while an image signal is usually a two or more dimensional function and may include more information of the source. Therefore, if the signal can be represented by such images as spectrogram, RASTA-PLP spectrum and auditory spectrum[3,4], more visualized features can be computed. In this paper, a novel method of feature extraction has been proposed which consists of visualization of acoustic signals and image feature extraction. Three types of underwater noise targets are used for the recognition experiment. The results have shown that the proposed method is effective.

---

## II. IMAGE REPRESENTATION OF ACOUSTIC FEATURES
An underwater noise signal \(x(p)\) can be divided into subframes \(x_{n}(m)\), in which \(p=0,1,2, \cdots, P-1\) is the number of sampling signal; \(P\) is the length; \(N\) is the length of the subframe; \(m=0,1,2, \cdots, N-1\).

The short-time Fourier transform of the subframe signal \(x_{n}(m)\) is:
$$
X\left(n, e^{j w}\right)=\sum x_{n}(m) w(n-m) e^{-j w w}
$$

Where \(w(m)\) is the window function. Then, the power spectrum function can be described as:
$$
Y(n, k)=|X(n, k)|^{2}=(X(n, k)) \times(\operatorname{conj}(X(n, k)))
$$

The spectrogram can be obtained according to the function \(Y(n, k)\), from which the following three types of visualized features can be extracted.

### A. GLCM-based Texture Feature Extraction
The feature parameters are extracted based on the gray level co-occurrence matrix which can reflect the comprehensive information of the image including direction, adjacent interval and change of amplitude.

Assuming that the total number of pixels in horizontal direction of the image is \(N_{x}\) and the number in vertical direction is \(N_{y}\), in order to reduce the amount of computation, the image grayscale is normalized. Suppose the highest grayscale level is \(N_{g}\), and:
$$
L_{x}=\left\{1,2, \cdots, N_{x}\right\}, L_{y}=\left\{1,2, \cdots, N_{y}\right\} \tag{3}
$$
$$
G=\left\{1,2, \cdots, N_{g}\right\}
$$

The image to be analyzed can be looked as the mapping from \(L_{x} ×L_{y}\) to \(G\). That means every point in the \(L_{x} × L_{y}\) is a corresponding grayscale of \(G\).

GLCM describes the possibility that a pair of \(d\) distance pixels each with grey level \(i\) and \(j\) in the direction of \(\theta\), and its elements can be denoted as \(P(i, j, d, \theta)\). Usually \(\theta\) takes \(0^{\circ}, 45^{\circ}, 90^{\circ}, 135^{\circ}\) as its value. The four directions of the GLCM are defined as follows:
$$
\begin{aligned}
P\left(i, j, 0^{\circ}, d\right)=&\#\left\{[(k, l),(m, n)] \in\left(L_{x} × L_{y}\right) \times\left(L_{x} × L_{y}\right),\right. \\
&\left.k-m=0,|l-n|=d, f(k, l)=i, f(m, n)=j\right\} \tag{5}
\end{aligned}
$$

$$
\begin{aligned}
P\left(i, j, 45^{\circ}, d\right)=&\#\{[(k, l),(m, n)] \in\left(L_{x} × L_{y}\right) \times\left(L_{x} × L_{y}\right), \\
&|k-m|=d,|l-n|=d, f(k, l)=i, f(m, n)=j\} \tag{6}
\end{aligned}
$$

$$
\begin{aligned}
P\left(i, j, 90^{\circ}, d\right)=&\#\left\{[(k, l),(m, n)] \in\left(L_{x} × L_{y}\right) \times\left(L_{x} × L_{y}\right),\right. \\
&|k-m|=d, l-n=0, f(k, l)=i, f(m, n)=j\} \tag{7}
\end{aligned}
$$

$$
\begin{aligned}
P\left(i, j, 135^{\circ}, d\right)= &\#\left\{[(k, l),(m, n)] \in\left(L_{x} × L_{y}\right) \times\left(L_{x} × L_{y}\right),\right. \\
&k-m=d, l-n=-d \text{ or } k-m=-d, \\
&l-n=d, f(k, l)=i, f(m, n)=j\} \tag{8}
\end{aligned}
$$

For an easy description of the statistics of the co-occurrence matrix, it is normalized as:
$$
P(i, j)=P(i, j) / R \tag{9}
$$

Where \(R\) represent the regularization constant, and when \(\theta=0^{\circ}\) or \(90^{\circ}\), \(R=2 N_{y}(N_{x}-1)\) and when \(\theta=45^{\circ}\) or \(135^{\circ}\), \(R=2(N_{y}-1)(N_{x}-1)\).

Four feature parameters in four directions \((0^{\circ}, 45^{\circ}, 90^{\circ}, 135^{\circ})\) are extracted including angular second moment(energy) \(f_{1}\), entropy \(f_{2}\), contrast degree \(f_{3}\), and correlation \(f_{4}\). For each parameter, to calculate the mean and mean square in 4 directions, we can obtain an 8-dimensional feature vector. The specific definitions of each feature parameters are listed below:
$$
f_{1}=\sum_{i=1}^{N_{g}} \sum_{j=1}^{N_{g}}[P(i, j)]^{2} \tag{10}
$$

$$
f_{2}=-\sum_{i=1}^{N_{g}} \sum_{j=1}^{N_{g}} P(i, j) \log [P(i, j)] \tag{11}
$$

$$
f_{3}=\sum_{n=0}^{N_{g}-1} n^{2}\left\{\sum_{\substack{i=1 \\|i-j|=n}}^{N_{g}} \sum_{j=1}^{N_{g}} P(i, j)\right\} \tag{12}
$$

$$
f_{4}=\left\{\sum_{i=1}^{N_{g}} \sum_{j=1}^{N_{g}} i × j × P(i, j)-\mu_{x} \mu_{y}\right\} / \delta_{x} \delta_{y} \tag{13}
$$

Where \(\mu_{x}\), \(\delta_{x}\) are the mean and mean square of \(\{P_{x}(i) ; i=1,2, \cdots, N_{g}\}\); \(\mu_{y}\), \(\delta_{y}\) are the mean and mean square of \(\{P_{y}(j) ; j=1,2, \cdots, N_{g}\}\).

### B. Texture Feature Extraction Based on the Grayscale-gradient Co-occurrence Matrix
Element \(H(x, y)\) of the grayscale-gradient co-occurrence matrix is defined as the number of pixels of the normalized grayscale image \(F(i, j)\) and its normalized gradient image \(G(i, j)\), in which the grayscale value is \(x\) and the gradient value is \(y\).

We can normalize the gray-gradient co-occurrence matrix and make the sum of its elements equals to 1, which can be showed as:
$$
\hat{H}(x, y)=\frac{H(x, y)}{\sum_{x=0}^{L-1} \sum_{y=0}^{L-1} H(x, y)} \tag{14}
$$

Since \(\sum_{x=0}^{L-1} \sum_{y=0}^{L_{g}-1} H(x, y)=N^{2}\), equation (14) can be rewritten as:
$$
\hat{H}(x, y)=\frac{H(x, y)}{N^{2}} \tag{15}
$$

Then the fifteen features of grayscale-gradient co-occurrence matrix are shown in Table I.

**TABLE I. FEATURES OF GRAYSCALE-GRADIENT CO-OCCURRENCE MATRIX**

| Feature | Definition |
| --- | --- |
| small | \(\sum_{x=0}^{L-1} \sum_{y=0}^{L-1} \hat{H}(x, y) \cdot(y+1)^{2}\) |
| large gradient | \(T_2=\sum_{x=0}^{L-1} \sum_{y=0}^{L_{g}-1} \hat{H}(x, y)\) |
| - | \(T_{3}=\sum_{x=0}^{L-1} x \sum_{y=0}^{L_{x}-1} \hat{H}(x, y)\) |
| Uneven grey grayscale | \(T_{i}=\frac{\sum_{j=1}^{n_{i}}\left[\sum_{i=1}^{n_{i}} \hat{u}_{i}(x, y)\right]}{\sum_{j=1}^{n_{i}} \sum_{i=1}^{n_{i}} \hat{u}_{i}(x, y)}\) |
| - | \(T_{5}=\left\{\sum_{i=0}^{L-1}\left(x-T_{n}\right)^{2} \sum_{j=0}^{L-1} \hat{H}(x, y)\right\}^{\frac{1}{2}}\) |
| - | \(T_{n}=\left\{\sum_{j=2}^{C_{n}-1}\left(y-T_{j}\right)^{2} \sum_{k=0}^{C_{n}-1} \hat{H}\left(x_{i}, y\right)\right\}^{\frac{1}{2}}\) |
| - | \(T_{13}=-\sum_{x=0}^{L+1} \sum_{y=0}^{L-1} \hat{H}(x, y) \log \hat{H}(x, y)\) |
| Inertia | - |

---

## III. COMBINATION OF PCA AND LDA FOR FEATURE SELECTION
Recognition experiments have been done based on the three types of features respectively and the results are not satisfactory. The results of the combination of these features are also not so good. That means effective features should be selected. So, we first compute the 30 features which includes 8 dimensional texture features based on GLCM, 15 dimensional texture features based on grayscale-gradient co-occurrence matrix, and 7 dimensional normalized central moments features. Then, some feature selection methods are applied including the PCA(principal component analysis) and LDA(linear discriminant analysis).

The PCA is a useful analysis tool based on linear transformation of high-dimensional data and compression after statistical analysis of the raw data. In this paper, it is used to reduce the feature dimension. The LDA can make the distance between the samples within the same class smaller, and the distance between ones that are not in the same class larger. This process does not include feature dimensionality reduction, and it maps the sample space to another, so that samples in the same class are more concentrated and the distance between samples of different classes becomes larger.

---

## IV. EXPERIMENTS
Noise signals of three underwater targets are created used as samples. Each type of sample signals last 5s with the sampling frequency 8kHz, and the resulting spectrograms are showed in Figure 1. In order to increase the number of the samples to get more information that has higher reference value, the visualized image of the noise signal is rotated in the range of \(0^{\circ} ~ 180^{\circ}\), and a new image can be obtained after each rotation. In that way, 18 image samples can be obtained. The total sample number of the three types of noise signals is 432, and half of them are used as training set and the other 216 are used as test set.

**Figure 1. Spectrogram of three types of noise signals**
(Three spectrogram plots with frame as horizontal axis and frequency as vertical axis)

The recognition results of the overall 30 visualized features are listed in Table II. It can be found that although the accurate rate of type C is perfect, the total rate is not high.

**TABLE II. RECOGNITION RATE OF 30 FEATURES**

| Type of signals | A | B | C | Total |
| --- | --- | --- | --- | --- |
| Recognition Rate(%) | 78.61 | 65.28 | 100 | 81.29 |

Figure 2 shows the comparison of the three types of feature selection methods which including PCA, LDA + PCA and PCA + LDA. It can be seen when the dimension is no less than 8, the method of PCA + LDA can achieve very good recognition rate which is higher than those of the other two methods. And the total recognition rate has been enhanced about 10 percent in contrast to that of the 30 features. With the feature dimensions reducing from 30 to 24, 20, 16, 12, 8, the recognition rate changes slowly. This means the selection method is very effective.

**Figure 2. Comparison of the results of the three methods**
(Line chart with feature dimensions as horizontal axis and recognition rate as vertical axis, curves of PCA, LDA+PCA, PCA+LDA)

---

## V. CONCLUSIONS
A novel method has been brought forward for the recognition of underwater targets by converting an acoustic signal into an image and then extracting and selecting the image features. A thirty features dataset has been obtained and the PCA and LDA are applied for the selection of effective visualized features. The simulation results have shown that when the feature dimensions are reduced (above a certain value, here it is 8) by PCA+LDA, the average recognition rate can be enhanced obviously.

In the next step, these selected features will be combined with those robust acoustic features such as loudness, MFCC, etc and the feasibility will be tested.

---

## ACKNOWLEDGEMENT
The project is supported by natural science basic research plan in Shaanxi Province of China (2012JM1010).

---

## REFERENCES
[1] C.Ioana,A.Quinquis and Y.Stephan,”Feature extraction from underwater signals using time-frequency warping operators”, IEEE Journal of Oceanic Engineering, vol.31, no.3, 2006,pp.628-645.
[2] Y.Wang,J.C.Sun and K.Chen,”Feature extraction of underwater targets based on psychoacoustic parameters”, Journal of Data Acquisition and Processing,vol.21,no.3, 2006,pp.313-317.
[3] J.R.He,X.Y.Zeng,“Method of extracting visualized features for sound signals”,Audio Engineering, vol.35, no.7,2011, pp.61-64.
[4] J.Dennis,”Spectrogram image feature for sound event classification in mismatched conditions”,.IEEE signal processing letters,vol.18,no.2, 2011,pp.130-133.

DOI: 10.1109/GCIS.2012.49
978-0-7695-4860-9/12 $26.00 © 2012 Crown Copyright

Authorized licensed use limited to: Nanjing Univ of Post & Telecommunications. Downloaded on March 27,2026 at 11:00:54 UTC from IEEE Xplore. Restrictions apply.