import os
from pathlib import Path

import numpy as np
import pylatex as pl

ROOT = str(Path(__file__).parent)
FOLDER = f"{ROOT}/tex"
os.makedirs(FOLDER, exist_ok=True)


def main():
    names = ["A", "B", "C", "D"]
    # values = [10, 12, 28, 21]
    # weights = [7, 4, 8, 9]
    values = [14, 56, 69, 76]
    weights = [10, 28, 23, 19]
    capacity = 60

    # Computations
    labels, v, w = prepare(names, values, weights)
    n0, updates = bound_and_branch(v, w, capacity)

    # Create tree with TikZ and generate PDF
    doc = create_tex(labels, v, w, n0, updates)
    doc.generate_pdf(clean_tex=False)


def prepare(names, values, weights):
    labels = np.array(names)
    w = np.array(weights)
    v = np.array(values)
    densities = v / w
    indexes = np.argsort(densities)[::-1]
    return labels[indexes], v[indexes], w[indexes]


def set_text(seq):
    if seq:
        seq.sort()
        return "\{" + ",".join(str(i + 1) for i in seq) + "\}"
    return "\\varnothing"


def vec_text(vec, dtype=int):
    return "(" + ",".join(format_scalar(f, dtype) for f in vec) + ")"


def format_scalar(scalar, dtype=int):
    dist = np.abs(scalar - dtype(scalar))
    return str(dtype(scalar)) if dist < 1e-6 else f"{scalar:.3f}"


class Node:
    def __init__(self, r, J0, J1, w, v, capacity, children=None, mother=False) -> None:
        self.i = 0
        self.r = r
        self.J0 = J0
        self.J1 = J1
        self.w = w
        self.v = v
        self.children = children
        self.capacity = capacity
        self.mother = mother
        self.sack_msg = ""
        self.msg = ""

    def to_tex(self):
        # Node text
        text = pl.Math(
            data=[
                r"{\color{blue}"
                + f"[{self.i+1}]\\ r = {self.r}"
                + "\\vspace{3pt}}\\newline\n",
                f"J_r^0 = {set_text(self.J0)}, J_r^1 = {set_text(self.J1)}\\newline\n",
                self.sack_msg,
            ],
            inline=True,
            escape=False,
        ).dumps()
        text += f"\\newline\n {self.msg}" if self.msg else ""

        child_text = ""
        if self.children is not None:
            child_text += "\n".join([c.to_tex() for c in self.children])

        # Node command
        if self.mother:
            text = "\\node{" + text + "}\n" + child_text + ";"
        else:
            text = "child { node{" + text + "}\n" + child_text + "}"
        return text

    def append(self, child):
        if self.children is None:
            self.children = [child]
        else:
            self.children.append(child)

    def add_children(self, children):
        for child in children:
            self.append(child)

    def add_message(self, msg):
        if self.msg:
            msg = ", " + msg
        self.msg += msg

    def depth(self):
        if self.children is None:
            return 1
        else:
            return 1 + np.max([c.depth() for c in self.children])

    def _create_sack_msg(self, xlb, xub, zlb, zub):
        self.sack_msg = "\\newline\n ".join(
            [
                "x_r^{UB} = " + vec_text(xub) + ",\\ z_r^{UB} = " + format_scalar(zub),
                "x_r^{LB} = " + vec_text(xlb) + ",\\ z_r^{LB} = " + format_scalar(zlb),
            ]
        )

    def calc(self, i=None):
        if i is not None:
            self.i = i
        idxs, next_idx = get_choices(self.J0, self.J1, self.w, self.capacity)

        # infeasable
        weight = np.sum(self.w[idxs])
        if weight > self.capacity:
            self.sack_msg = "\\newline"
            return 4 * [None]

        xlb = np.zeros_like(self.v, float)
        xlb[idxs] = 1
        zlb = np.dot(xlb, self.v)

        xub = xlb.copy()
        if next_idx is not None:
            xub[next_idx] = (self.capacity - weight) / self.w[next_idx]
        zub = np.dot(xub, self.v)

        self._create_sack_msg(xlb, xub, zlb, zub)
        return next_idx, xlb, zlb, zub


def create_tex(labels, values, weights, mother_node, updates, width=6.5):
    # Initialize document
    doc = pl.Document(
        f"{FOLDER}/branch_and_bound",
        documentclass="scrarticle",
        document_options=["a1paper", "landscape"],
        font_size="Large",
        geometry_options={
            "left": "1.5cm",
            "right": "1.5cm",
            "bottom": "1.5cm",
            "top": "1.5cm",
        },
    )
    doc.packages.append(pl.Package("amssymb"))
    doc.packages.append(pl.Package("array"))
    doc.packages.append(pl.Package("tikz"))
    doc.packages.append(pl.Command("usetikzlibrary", "trees"))
    doc.preamble.append(pl.UnsafeCommand("newcolumntype", ["M", ">{$}c<{$}"]))

    # Create sorted data table
    n_labels = len(labels)
    rows = [
        pl.table.MultiColumn(n_labels + 1, data=pl.Command("textbf", "Data")),
        r"\\",
        "Item & " + " & ".join(label for label in labels) + r"\\\hline",
        "Value & " + " & ".join(str(v) for v in values) + r"\\",
        "Weights & " + " & ".join(str(v) for v in weights) + r"\\",
    ]
    table = pl.Tabular("l|" + n_labels * "c", rows, "t")
    table.escape = False
    doc.append(table)
    doc.append(pl.Command("hspace", "2cm"))

    # Create update table
    # doc.append(pl.ColumnType("M", "c", "$", parameters=0))
    rows = [
        pl.table.MultiColumn(
            len(updates[0]), data=pl.Command("textbf", "Global Updates")
        ),
        r"\\",
        "i & r & x^{LB} & z^{LB}" + r"\\\hline",
    ]
    rows += [f"{i} & {r} & {vec_text(x)} & {int(z)} \\\\" for i, r, x, z in updates]
    uptable = pl.Tabular("MMMM", rows, "t")
    uptable.escape = False
    doc.append(uptable)
    doc.append(pl.Command("small"))

    # Define TikZ options
    depth = mother_node.depth() - 2
    max_dist = (width + 1.0) * 2 ** depth
    options = [
        "level distance=3cm",
        f"sibling distance={max_dist}cm",
        "every node/.style={shape=rectangle, draw, align=center, "
        f"text width={width}cm" + ", align=left}",
    ]
    options += [
        f"level {2+level}/.style="
        "{sibling distance=" + str(max_dist / (2 ** (level + 1))) + "cm}"
        for level in range(depth)
    ]

    # Draw Tikz tree graph
    with doc.create(pl.Figure(position="t")) as _:
        tikz = pl.TikZ(data=[mother_node.to_tex()], options=pl.TikZOptions(options))
        tikz.escape = False
        doc.append(tikz)
    return doc


def create_node(i, r, J0, J1, mother, **kwargs):
    return {"i": i, "r": r, "J0": J0, "J1": J1, "mother": mother, **kwargs}


def get_choices(j0, j1, w, capacity):
    choices = list(set(range(len(w))) - set(j0) - set(j1))
    choices.sort()
    idxs = j1.copy()
    while choices:
        c = choices[0]
        prop = idxs + [c]
        if np.sum(w[prop]) <= capacity:
            idxs = prop.copy()
        else:
            break
        choices = choices[1:]
    c = None
    if choices:
        c = choices[0]
    return idxs, c


def bound_and_branch(v, w, capacity):
    i = 0
    r = 1
    prev_idxs = []
    nodes = []
    n0 = Node(0, [], [], w, v, capacity, mother=True)
    nodes.append(n0)
    cn = n0
    glb = np.sum(w)
    updates = []
    while True:
        next_idx, xlb, zlb, zub = cn.calc(i)
        i += 1
        if zlb is None:
            cn.add_message("Pruning by infeasability")
            if not prev_idxs:
                break
            cn = nodes[prev_idxs.pop(-1)]
            continue

        # Update globals
        if zlb > glb:
            glb = zlb
            updates.append([i, cn.r, xlb, glb])
            cn.add_message("Global update")

        dom = zub <= glb  # check for domincance
        opt = zlb == zub  # check for optimality
        if dom | opt:
            msg = "Pruning by " + ", ".join(
                t for t, flag in zip(["dominance", "optimality"], [dom, opt]) if flag
            )
            cn.add_message(msg)
            if not prev_idxs:
                break
            cn = nodes[prev_idxs.pop(-1)]
            continue
        else:  # Branch
            ln = Node(r, cn.J0 + [next_idx], [], w, v, capacity)
            rn = Node(r + 1, [], cn.J1 + [next_idx], w, v, capacity)
            prev_idxs.append(r)
            nodes.extend([ln, rn])
            cn.add_children([ln, rn])
            r += 2
            cn = nodes[-1]
    return n0, updates


if __name__ == "__main__":
    main()
