"""Regenerate clipped_blocks.json for clipped_finite.py (Proposition 8.10).

Greedy: starting at n = 194, take the widest block [n, n + w - 1] (w doubled
while it still passes, then extended by bisection) whose ball enclosure of the
normalized ratio R lies below THRESHOLD.  The checker does not trust this search; it
re-evaluates every block.

Run: python3 make_blocks.py   (needs python-flint; a few seconds)
"""
import json

from clipped_finite import BLOCKS, THRESHOLD, ratio


def ok(n1, n2):
    return ratio(n1, n2) < THRESHOLD


def main(lo=194, hi=3998):
    blocks, n = [], lo
    while n <= hi:
        assert ok(n, n), n
        w = 1
        while n + 2 * w - 1 <= hi and ok(n, n + 2 * w - 1):
            w *= 2
        good, bad = w, min(2 * w, hi - n + 1) + 1  # good passes, bad fails or overshoots
        while bad - good > 1:
            mid = (good + bad) // 2
            if ok(n, n + mid - 1):
                good = mid
            else:
                bad = mid
        blocks.append([n, n + good - 1])
        n += good
    BLOCKS.write_text(json.dumps({"range": [lo, hi], "blocks": blocks}) + "\n")
    print(len(blocks), "blocks")


if __name__ == "__main__":
    main()
