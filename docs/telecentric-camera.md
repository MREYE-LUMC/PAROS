# Derivation of the magnification for a telecentric camera

## Definitions

$M = \begin{bmatrix} A & B \\ C & D \end{bmatrix}$ is the ray transfer matrix of the eye model.
Since the object space is vitreous and the image space is air, $\det M = AD - BC = n_{vit}$.
A ray is defined as $(y, u)$ with $y$ the height of the ray and $u$ its angle with the optical axis.

## Telecentricity

A telecentric camera linearly converts the eccentricity ($\theta$) of an incoming ray into a position on the sensor:

$$x_{px} = k \theta$$

with $k$ constant over the full focus range.

## Derivation of the magnification

The chief ray leaves the retina at a height $y$ and unknown angle $u$.
At the entrance pupil plane, reached by propagating over a distance $p$, its height must vanish:

$$(A + pC) y + (B + pD) u = 0 \Rightarrow u = -\frac{A + pC}{B + pD} y$$

Its angle in air after the cornea is $\theta = Cy + Du$, so

$$\theta = \frac{C (B + pD) - D (A + pC)}{B + pD} y = \frac{AD - BC}{B + pD} y = -\frac{n_{vit}}{B + pD} y$$

Let $s$ be the retinal distance subtended per radian of camera eccentricity:

$$s = \frac{\mathrm{d} y}{\mathrm{d} \theta} = -\frac{B + pD}{n_{vit}}$$

The true retinal distance is then

$$d_{true} = \frac{x_{px}}{k} \frac{B + pD}{n_{vit}}$$

And the magnification is

$$m_{[px/mm]} = \frac{k\ n_{vit}}{10^3 (B + pD)}.$$

## Derivation of the entrance pupil distance

The magnification depends on the distance $p$ between the cornea and the entrance pupil.
Since the pupil is located between the posterior and anterior segments, the ray transfer matrix must be split into a posterior and anterior part: $M = M_{ant} M_{post}$.
Let $(y_S, u_S)$ be a point in the physical pupil.
Since we are tracing chief rays, $y_S = 0$ and the ray crosses the pupil plane in $(0, u_S)$.
At the entrance pupil plane, the ray is 

$$
\begin{bmatrix} y_E \\ u_E \end{bmatrix} = 
\begin{bmatrix} 1 & p \\ 0 & 1 \end{bmatrix} M_{ant} \begin{bmatrix} 0 \\ u_S \end{bmatrix} =
\begin{bmatrix} B_{ant} u_S + p D_{ant} u_S \\ D_{ant} u_S \end{bmatrix}
$$

Since chief rays are defined to pass through the entrance pupil, $y_E = 0$, so 

$$(B_{ant} + pD_{ant}) u_S = 0 \Rightarrow p = -\frac{B_{ant}}{D_{ant}}.$$
