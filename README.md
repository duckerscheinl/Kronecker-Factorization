# Tensor-product decomposition of a two-site operator

Given an operator $M$ acting on two sites, we want to write it as a sum of tensor
products of single-site operators,

```math
M \;=\; \sum_{k} A_k \otimes B_k ,
```

with as few terms as possible. This is the *operator Schmidt decomposition*, and it is
the elementary step behind building an MPO from a dense operator: the number of terms
needed at a bond is exactly the MPO bond dimension there.

The trick is that the decomposition is not obtained from the spectrum of $M$, but from a
**reshuffling of its indices** followed by an SVD.

## The construction

![Index realignment and SVD, worked through for sigma_x tensor sigma_z](figure.png)

The four colours are the four $2\times2$ blocks of $M$: each block corresponds to one
value of the left-site index pair $(r_{\mathrm L}, c_{\mathrm L})$ and becomes one row of $R$.

Read the sketch in four steps.

**1. Start from the dense matrix.** For $\sigma_x \otimes \sigma_z$,

```math
\sigma_x \otimes \sigma_z =
\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \otimes
\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} =
\begin{pmatrix}
0 & 0 & 1 & 0 \\
0 & 0 & 0 & -1 \\
1 & 0 & 0 & 0 \\
0 & -1 & 0 & 0
\end{pmatrix}.
```

**2. Read the composite indices in binary.** Each row index is a ket $|r_{\mathrm L} r_{\mathrm R}\rangle$
and each column index a bra $\langle c_{\mathrm L} c_{\mathrm R}|$, the left bit belonging to the
first site and the right bit to the second:

```math
M_{(r_{\mathrm L} r_{\mathrm R}),\,(c_{\mathrm L} c_{\mathrm R})} = A_{r_{\mathrm L} c_{\mathrm L}} \, B_{r_{\mathrm R} c_{\mathrm R}} .
```

**3. Realign.** Regroup the four bits so that the two indices of the *same site* sit
together instead of the two indices of the same type:

```math
R_{(r_{\mathrm L} c_{\mathrm L}),\,(r_{\mathrm R} c_{\mathrm R})} \;:=\; M_{(r_{\mathrm L} r_{\mathrm R}),\,(c_{\mathrm L} c_{\mathrm R})} .
```

In the sketch this is the "lookup representation": the row label $|a{\times}b|$ stands for the
basis operator $|a\rangle\langle b|$ of the left site, the column label for the same on the
right site. The coloured arrows track where each of the 16 entries is sent. The result is

```math
R =
\begin{pmatrix}
0 & 0 & 0 & 0 \\
1 & 0 & 0 & -1 \\
1 & 0 & 0 & -1 \\
0 & 0 & 0 & 0
\end{pmatrix},
```

which is nothing but $\mathrm{vec}(\sigma_x)\,\mathrm{vec}(\sigma_z)^{\mathsf T}$.

**4. SVD.** $R = \sum_k s_k\, u_k v_k^{\dagger}$, and each singular vector reshapes back into a
$2\times2$ single-site operator:

```math
A_k = \sqrt{s_k}\,\mathrm{vec}^{-1}(u_k), \qquad B_k = \sqrt{s_k}\,\mathrm{vec}^{-1}(v_k^{\dagger}).
```

Here $R$ has rank 1, so a single term survives, and it returns the factors we started with:

```math
u \;\propto\; (0,1,1,0)^{\mathsf T} \;\hat{=}\; \sigma_x, \qquad
v^{\dagger} \;\propto\; (1,0,0,-1) \;\hat{=}\; \sigma_z .
```

The rank of $R$ — not of $M$ — is the operator Schmidt rank. For $\sigma_x \otimes \sigma_z$ it is 1;
for $\sigma_x \otimes \sigma_z + \sigma_z \otimes \sigma_x$ it is 2, and so on.

## Reference implementation

```python
import numpy as np

def realign(M, dA=2, dB=2):
    """M[(iA iB), (jA jB)]  ->  R[(iA jA), (iB jB)]"""
    return (M.reshape(dA, dB, dA, dB)     # iA iB jA jB
             .transpose(0, 2, 1, 3)       # iA jA iB jB
             .reshape(dA * dA, dB * dB))

def tensor_terms(M, dA=2, dB=2, tol=1e-12):
    """Return lists A, B with  M = sum_k kron(A[k], B[k])."""
    U, s, Vh = np.linalg.svd(realign(M, dA, dB))
    r = int((s > tol * max(s[0], 1.0)).sum())
    A = [np.sqrt(s[k]) * U[:, k].reshape(dA, dA) for k in range(r)]
    B = [np.sqrt(s[k]) * Vh[k, :].reshape(dB, dB) for k in range(r)]
    return A, B
```

```python
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])

A, B = tensor_terms(np.kron(sx, sz))
len(A)                                            # -> 1, the Schmidt rank
np.allclose(sum(np.kron(a, b) for a, b in zip(A, B)), np.kron(sx, sz))   # -> True
```

## Notes and conventions

- **Index ordering.** Big-endian throughout: the composite index is $2 r_{\mathrm L} + r_{\mathrm R}$, so the
  left bit is the slow index and belongs to the first tensor factor. This matches
  `numpy.kron` and the C ordering of `reshape`; a column-major convention swaps the roles
  of the two bits and transposes $R$.
- **Normalisation.** The sketch writes the middle factor as $1$ and keeps the singular
  vectors unnormalised. With unit-norm $u$ and $v$ the singular value is
  $\lVert\mathrm{vec}\,\sigma_x\rVert \cdot \lVert\mathrm{vec}\,\sigma_z\rVert = \sqrt{2}\cdot\sqrt{2} = 2$,
  and $u = (0,1,1,0)^{\mathsf T}/\sqrt 2$, $v = (1,0,0,-1)^{\mathsf T}/\sqrt 2$. Splitting $\sqrt{s_k}$ onto
  each side, as the code does, is the symmetric choice; how the scalar is distributed
  between $A_k$ and $B_k$ is pure gauge.
- **Sign/phase gauge.** The pair $(A_k, B_k)$ is fixed only up to $(e^{i\theta}A_k, e^{-i\theta}B_k)$.
  LAPACK will happily return $(-\sigma_x, -\sigma_z)$ instead of $(\sigma_x, \sigma_z)$; the product is the same.
- **Realignment is a permutation, not a similarity transform.** It does not preserve the
  spectrum of $M$, only its Frobenius norm, $\lVert R\rVert_F = \lVert M\rVert_F = \sqrt{\sum_k s_k^2}$.
  So the singular values of $R$ say nothing about the eigenvalues of $M$ — they measure how
  entangled the *operator* is across the bond.
- **Complex entries.** The SVD hands back $v_k^{\dagger}$, i.e. the conjugated row of `Vh`. Reshape
  that row directly, as above; do not conjugate it a second time.
- **More than two sites.** Apply the same reshuffle-and-SVD at each bond, sweeping
  left to right and carrying the remainder $s\,V^{\dagger}$ into the next site. The retained
  singular values at each cut are the MPO bond dimensions, and truncating small ones
  gives a controlled approximate MPO.

## Files

| File | Description |
|---|---|
| `figure.png` | Diagram above, raster (300 dpi) |
| `figure.svg` | Same diagram, vector |
| `figure.pdf` | Same diagram, vector — drop straight into LaTeX with `\includegraphics` |
| `figure.tex` | TikZ source (`standalone`, compiles with `pdflatex`) |
| `sketch.png` | Original hand-drawn derivation (cleaned scan) |

The hand-drawn original, for reference:

![Original handwritten derivation](sketch.png)
