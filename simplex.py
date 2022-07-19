import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pylatex as pl
from intvalpy import lineqs

ROOT = str(Path(__file__).parent)
FOLDER = f"{ROOT}/tex"
os.makedirs(FOLDER, exist_ok=True)


def main():
    pdf = True
    precision = 3

    c = [2, -6]
    A = [
        [-3, 1],
        [-1, 2],
        [2, 3],
        [6, -4],
        [-4, -2],
    ]
    b = [-4, 7, 28, 32, -12]
    s = [2, 2]

    A, b, c = to_arrays(A, b, c)

    img_file = plot_solution_space(A, b, c)

    doc = None
    if pdf:
        doc = pl.Document(
            f"{FOLDER}/simplex",
            data=[pl.Section("Simplex")],
            geometry_options={
                "head": "2cm",
                "left": "1.5cm",
                "bottom": "2cm",
                "right": "1.5cm",
                "includeheadfoot": True,
            },
        )
        doc.append(
            pl.Math(data=["A =", Matrix(A), ", b =", Matrix(b), ", c =", Matrix(c)])
        )
        with doc.create(pl.Figure(position="h")) as fig:
            fig.add_image(img_file, width=pl.NoEscape(r"\linewidth"))

    simplex(A, b, c, s, doc=doc, precision=precision)

    if doc is not None:
        doc.generate_pdf(clean_tex=False)
    plt.show()


class Matrix(pl.basic.Environment):
    """A class representing a matrix."""

    packages = [pl.Package("amsmath")]

    _repr_attributes_mapping = {
        "alignment": "arguments",
    }

    def __init__(self, matrix, *, mtype="b", alignment="r", precision=5):
        r"""
        Args
        ----
        matrix: `numpy.ndarray` instance
            The matrix to display
        mtype: str
            What kind of brackets are used around the matrix. The different
            options and their corresponding brackets are:
            p = ( ), b = [ ], B = { }, v = \| \|, V = \|\| \|\|
        alignment: str
            How to align the content of the cells in the matrix. This is ``c``
            by default.

        References
        ----------
        * https://en.wikibooks.org/wiki/LaTeX/Mathematics#Matrices_and_arrays
        """
        self.matrix = matrix

        self.latex_name = mtype + "matrix"
        self._mtype = mtype
        self.precision = precision
        if alignment is not None:
            self.latex_name += "*"
            self.packages.append(pl.Package("mathtools"))

        super().__init__(options=alignment)

    def dumps_content(self):
        """Return a string representing the matrix in LaTeX syntax.

        Returns
        -------
        str
        """
        string = ""
        shape = self.matrix.shape

        for (y, x), value in np.ndenumerate(self.matrix):
            if x:
                string += "&"
            v = np.round(value, self.precision)
            if np.abs(v) <= 10 ** (-self.precision - 1):
                v = np.abs(v)
            s = str(v)
            if s == "inf":
                s = r"\infty"
            string += s

            if x == shape[1] - 1 and y != shape[0] - 1:
                string += r"\\" + "%\n"

        super().dumps_content()

        return string


def to_arrays(A, b, c):
    return np.array(A), np.expand_dims(b, -1), np.expand_dims(c, 0)


def get_boundaries(vals, offset=0.1):
    lb = min(-offset, np.min(vals) - offset)
    return np.array([lb, np.max(vals) + offset])


def get_ticks(lims):
    lb, ub = np.ceil(lims)
    return np.arange(lb, ub)


def get_edge_indexes(A, b, v):
    v = np.array(v)
    dev = np.abs(A @ v[:, None] - b)
    mask = np.squeeze(dev) < 1e-6
    indexes = np.arange(len(A))[mask]
    return indexes


def plot_solution_space(A, b, c, offset=0.1, doc=None):
    file = f"{FOLDER}/plot.pdf"
    vertices = lineqs(-A, -b, show=False)
    x, y = vertices[:, 0], vertices[:, 1]
    xlims = get_boundaries(x, offset)
    ylims = get_boundaries(y, offset)
    cn = np.squeeze(c / np.linalg.norm(c))
    best_idx = np.argmax(vertices @ c.T)

    fig, ax = plt.subplots(constrained_layout=True)
    for ai, bi in zip(A, b):
        if np.abs(ai[1]) >= 1e-3:
            x1 = xlims.copy()
            x2 = (bi[0] - ai[0] * x1) / ai[1]
        else:
            x2 = ylims.copy()
            x1 = bi[0] * np.ones_like(x2)
        ax.plot(x1, x2, label=f"${ai[0]}\\, x_1 + {ai[1]}\\, x_2 \\leq {bi[0]} $")
    ax.arrow(0, 0, *cn, width=0.01, edgecolor="g", facecolor="g", label="$c$")
    ax.fill(x, y, color="k", alpha=0.25)
    ax.plot(x, y, "mo")
    ax.plot(x[best_idx], y[best_idx], "go", label="maximum")

    ax.legend()
    ax.grid(1)
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    ax.set_xticks(get_ticks(xlims))
    ax.set_yticks(get_ticks(ylims))
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    fig.savefig(file)
    return file


def simplex(A, b, c, s, doc=None, precision=3):
    pdf = doc is not None
    B = get_edge_indexes(A, b, s)
    count = 1
    v = np.array(s, float)[:, None]
    while True:
        AB = A[B]
        # bB = b[B]
        Ainv = np.linalg.inv(AB)
        # v = Ainv @ bB
        u = (c @ Ainv).T
        i = np.argmin(u)
        d = -Ainv[:, i][:, None]

        if pdf:
            doc.append(pl.Subsection(f"Iteration {count}", False))
            doc.append(
                pl.Math(
                    data=[
                        "v =",
                        Matrix(v.copy(), precision=precision),
                        ", B =",
                        Matrix(B[None] + 1, precision=precision),
                        ", A_B =",
                        Matrix(AB, precision=precision),
                        f", c v ={np.squeeze(c @ v):.3f}",
                    ],
                    escape=False,
                )
            )
            doc.append(
                pl.Math(
                    data=[
                        "A_B^{-1} =",
                        Matrix(Ainv, precision=precision),
                        ", u = c A_B^{-1} =",
                        Matrix(u, precision=precision),
                    ],
                    escape=False,
                )
            )

        if u[i, 0] >= 0:
            doc.extend(
                [
                    "Stop because ",
                    pl.Math(
                        data=[r"u_i \geq 0, \forall i \in I"], inline=True, escape=False
                    ),
                    ", Solution: ",
                    pl.Math(
                        data=[r"\Rightarrow v =", Matrix(v, precision=precision)],
                        inline=True,
                        escape=False,
                    ),
                ]
            )
            break
        # j = B[i]

        Av = A @ v
        Ad = A @ d
        mask = np.squeeze(Ad > 1e-12)
        lambdas = np.full(A.shape[0], np.infty)
        lambdas[mask] = np.squeeze((b[mask] - A[mask] @ v) / Ad[mask])

        k = np.argmin(lambdas)
        v += lambdas[k] * d
        B[i] = k
        B = np.sort(B)

        doc.append(
            pl.Math(
                data=[
                    "d = ",
                    Matrix(d, precision=precision),
                    ", A v =",
                    Matrix(Av, precision=precision),
                    ", A d =",
                    Matrix(Ad, precision=precision),
                    r", \lambda =",
                    Matrix(lambdas[:, None], precision=precision),
                    f", \lambda^* = {np.min(lambdas):.3f}",
                ],
                escape=False,
            )
        )
        doc.extend(
            [
                "The minimum is obtained at index ",
                pl.Math(inline=True, data=f"k = {k+1}", escape=False),
            ]
        )
        count += 1
    return v[i]


if __name__ == "__main__":
    main()
