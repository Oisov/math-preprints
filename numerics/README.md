# Numerics

Scripts that check the computer-assisted statements in the drafts of this
repository.

- `book/`: the book draft *Where Nehari's Theorem Survives* (v4). Run
  `python3 book/run_all.py`; see `book/README.md` for the table of statements,
  scripts and requirements.
- `opa-zeros/`: the paper *Zeros of optimal polynomial approximants in H^p*.
  Run `python3 opa-zeros/run_all.py`; see `opa-zeros/README.md`.

Where a statement rests on a computation, the scripts here are the complete
check: exact rational or interval (Arb) arithmetic, except where a README says
otherwise.
