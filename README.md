# Panorama_generation
# Explanation of Solution

## Interest Point Detection Using SIFT

We first begin by identifying the corresponding interest points in all sets of 2 adjacent images. These interest points give us a starting set of points that might correspond to each other in the two images. To find these interest points, I use the SIFT algorithm, and to match the interest points with each other, I use the Brute-Force (BF) matcher. These matched interest points are then refined using RANSAC.

## RANSAC

The Random Sample Consensus (RANSAC) algorithm uses all the interest points in two images to first estimate a homography between the two images and then finally refines that estimate into a final homography. RANSAC does this by sampling 'n' sets of random interest points between two images and using them to estimate a homography. This estimated homography is then used to transform all the interest points from the first image into interest points on the second image. The distance between the actual and estimated interest point on the second image is calculated. If this distance is below a certain threshold $\delta $, this point is considered an inlier and supports the estimated homography. On the other hand, if the distance is greater than $\delta $, the point is considered an outlier and does not support the homography.

The set of all inlier points is called the inlier set. Likewise, the set of all outlier points is called the outlier set. The larger the inlier set, the greater is the support for that specific estimated homography. RANSAC keeps sampling random sets of 'n' interest points until it is able to find a homography that is supported by an inlier set larger than a set threshold. The threshold for the minimum size of the inlier set is called $M $.

### The parameters used in RANSAC are below:
- Number of random correspondences used to estimate homography $n$: 6  
- Decision threshold used to construct inlier set $\delta$: 3  
- Probability that a random correspondence is an outlier $\epsilon$: 0.5  
- Probability that at least one trial is free of outliers $p$: 0.99  
- Number of trials conducted to achieve above probability $N$:  
  $N = \frac{\ln(1 - p)}{\ln(1 - (1 - \epsilon)^n)}$
- Minimum value of inlier set $M$:  
  $M = (1 - \epsilon) \times \text{total number of interest points}$

## Linear Least Squares

For the estimation of the homography in RANSAC, I use linear least squares. We know that our goal is to estimate a homography that relates the interest points in one image to the corresponding interest points in the second image. 

So, for the set of interest points $(X, X')$, we want to estimate $H$such that $X' = HX$.  
This gives us:
$X' \times HX = 0$

Where:

$$
X' = 
\begin{bmatrix}
x' \\
y' \\
w'
\end{bmatrix}
$$

$$
X =
\begin{bmatrix}
x \\
y \\
w
\end{bmatrix}
$$

$$
H =
\begin{bmatrix}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31} & h_{32} & h_{33}
\end{bmatrix}
$$

For one correspondence, this vector product can be simplified into:

$$
0 \cdot h^1 - w' X^T h^2 + y' X^T h^3 = 0
$$

$$
w' X^T h^1 + 0 \cdot h^2 - x' X^T h^3 = 0
$$

$$
-y' X^T h^1 + x' X^T h^2 + 0 \cdot h^3 = 0
$$

Now, given that we have $n$correspondences and setting $h_{33}$ to 1, we can construct the following matrices:

$$
\begin{bmatrix}
0 & 0 & 0 & -w'x & -w'y & -w'w & y'x & y'y \\
w'x & w'y & w'w & 0 & 0 & 0 & -x'x & -x'y \\
. & . & . & . & . & . & . & . \\
. & . & . & . & . & . & . & .
\end{bmatrix}
$$

$$
\begin{bmatrix}
h_{11} \\
h_{12} \\
h_{13} \\
h_{21} \\
h_{22} \\
h_{23} \\
h_{31} \\
h_{32}
\end{bmatrix}
$$

$$
\begin{bmatrix}
-y'w \\
x'w \\
. \\
.
\end{bmatrix}
$$

This can be represented as:

$$
Ah = b
$$

where $A$ is a $2n \times 8$ matrix, $h$ is an 8-element vector with our unknowns, and $b$ is a $2n$ -element vector.

The solution to this equation is:

$$
h = (A^T A)^{-1} A^T b
$$

Here, $A^T A^{-1} A^T$ is called the pseudo-inverse of matrix $A$. Using this equation and the pseudo-inverse, we can determine our linear least squares estimation of the homography.

## Levenberg-Marquardt (LM) Algorithm

The Levenberg-Marquardt (LM) method of non-linear least squares is a combination of Gradient Descent and Gauss-Newton methods. For initial guesses that are far from the function minimum, LM behaves like Gradient Descent. Once the estimate is sufficiently close to the true minimum, LM behaves like Gauss-Newton and makes a leap to the final answer.

For a guess $p$, LM begins by calculating the cost function below:

$$
C(p) = \| X - f(p) \|^2
$$

Where:

$$
X =
\begin{bmatrix}
x_1' \\
y_1' \\
x_2' \\
. \\
. \\
y_n'
\end{bmatrix}
$$

and $f(p)$ is a function that calculates the points given a homography. For a point $(x_i, y_i)$, the function has the form:

$$
f_1^i = \frac{h_{11} x_i + h_{12} y_i + h_{13}}{h_{31} x_i + h_{32} y_i + h_{33}}
$$

$$
f_2^i = \frac{h_{21} x_i + h_{22} y_i + h_{23}}{h_{31} x_i + h_{32} y_i + h_{33}}
$$

To implement this, I used the `least_squares` function in the SciPy library. The cost function was implemented using the above formulas. The process of least squares minimization was handled by SciPy, and the final output from the function was reshaped into a $3 \times 3$ matrix to form the homography.

## Generating the Panorama

The first step here is to estimate the bounds of the final output so that we can create a blank canvas of the appropriate size. To do this, I transform all the corners of all the images using their corresponding homographies. All these corner points are then compared to find the 4 most extreme corners. These are the 4 points that form the range of the final output image.

Next, since the LM output gives us the homographies between adjacent images, we need to convert all the homographies to a common frame. For this, the common frame I chose was the frame of the middle-most image.

The homographies for the images before the middle image are obtained by multiplying all the homographies that lie between it and the middle image. The homographies for the images after the middle image are obtained by taking the inverse of the multiplication of all the homographies that lie between the middle image and itself. This gives us all the homographies in the common frame of the middle image. Finally, all these homographies are applied to the corresponding images, and they are all warped and projected onto the final image.
