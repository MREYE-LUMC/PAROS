# Derivation of the refraction for a paraxial eye model

PAROS calculates the refraction of an eye by placing a virtual corrective lens 14 mm in front of the cornea, and calculating the power required to put the image of the retina at infinity.
This power is equal to the vergence of the rays 14 mm in front of the cornea, which is calculated using the ray transfer matrix of the eye model.

Let $M = \begin{bmatrix} A & B \\ C & D \end{bmatrix}$ be the ray transfer matrix of the eye model.
A ray starts on the retina at $(y, u)$ with $y$ the height of the ray and $u$ its angle with the optical axis.
After propagation through the eye and a distance $d$ in front of the cornea, the ray is transformed to

$$
\begin{bmatrix} y' \\ u' \end{bmatrix} =
\begin{bmatrix} 1 & d \\ 0 & 1 \end{bmatrix} M \begin{bmatrix} y \\ u \end{bmatrix}.
$$

For a central ray, $y = 0$:

$$
\begin{bmatrix} y' \\ u' \end{bmatrix} =
\begin{bmatrix} 1 & d \\ 0 & 1 \end{bmatrix} M \begin{bmatrix} 0 \\ u \end{bmatrix} =
\begin{bmatrix} B u + d D u \\ D u \end{bmatrix}.
$$

The vergence is defined as $V = n / d$ with $n$ the refractive index of the medium and $d$ the distance to the source.
In the paraxial approximation, $\tan \theta \approx \theta$ and $\tan \theta = y / d$, so $d = y / \theta$ and $V = n \theta / y$.

Using this definition, the vergence of the rays 14 mm in front of the cornea is (assuming the refractive index of air $n = 1$):

$$V = n \frac{u'}{y'} = \frac{n D u}{B u + d D u} = \frac{n D}{B + d D} = \frac{D}{B + d D}.$$