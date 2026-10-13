"""Regenerate convex_tree.json for unimodality.py (Lemma 6.22).

Depth-first adaptive subdivision of each p-range of the box (alpha, p): a box
is a leaf ("L") when the mean value form proves F > 0, is marked "O" when it
misses the image of q in [1e-40, 8], and is otherwise split in alpha ("A") or
in log p ("P"), whichever gradient term is wider.  The flags are stored in
depth-first order; unimodality.py replays the same tree and re-evaluates
every leaf, so it does not trust this search.

Run: python3 make_convex_tree.py   (needs python-flint; a few seconds)
"""
import json
import math

from flint import ctx

from unimodality import HERE, box, children, outside, ranges

ctx.prec = 64
A0, A1, rngs = ranges()
tree = []
for p0, p1, deflated in rngs:
    flags = []
    stack = [(A0, A1, p0, p1)]
    while stack:
        node = stack.pop()
        if outside(*node):
            flags.append("O")
            continue
        v, ea, eb = box(*node, deflated)
        if v > 0:
            flags.append("L")
            continue
        a1, a2, q1, q2 = node
        if ea is not None and ea.is_finite() and eb.is_finite():
            fl = "A" if ea.rad() > eb.rad() else "P"
        else:
            fl = "A" if (a2 - a1) > 0.5 * math.log(q2 / q1) else "P"
        flags.append(fl)
        stack += children(node, fl)
        assert len(flags) < 10**6
    tree.append("".join(flags))
    print(p0, p1, "leaves", flags.count("L"), "outside", flags.count("O"))
(HERE / "convex_tree.json").write_text(json.dumps({"flags": tree}) + "\n")
