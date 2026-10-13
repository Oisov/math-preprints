# Numerics

Scripts that check the computer-assisted statements in the drafts of this
repository.

- `book/`: the book draft *Where Nehari's Theorem Survives* (v4). Run
  `python3 book/run_all.py`; see `book/README.md` for the table of statements,
  scripts and requirements.
- `opa-zeros/`: the paper *Zeros of optimal polynomial approximants in H^p*.
  Run `python3 opa-zeros/run_all.py`; see `opa-zeros/README.md`.
- `hilbert-points/`: the note *Two remarks on Hilbert points in Hardy spaces*.
  `python3 hilbert-points/equal_modulus_certificate.py` certifies, in exact
  integer interval arithmetic (standard library only, a few seconds), that
  D(1/5) < 0 in Theorem 2.
- `paley-wiener/`: the note *Remarks on three conjectures on point evaluation
  in Paley–Wiener spaces*. `python3 paley-wiener/H-PW-2-certificate.py`
  certifies Theorem 1 (Conjecture 9.2 of Brevig–Chirre–Ortega-Cerdà–Seip with
  A = 7/100) in Arb ball arithmetic; pass `0.06` for A = 3/50. Requires
  python-flint 0.9.0; about a minute.
- `composition-sharpness/`: the note *Remarks on three sharpness questions for
  composition operators on spaces of Dirichlet series*.
  `python3 composition-sharpness/jets.py` checks, in exact rational-function
  arithmetic (sympy, about a second), the Taylor identities of Lemma 7.

Where a statement rests on a computation, the scripts here are the complete
check: exact rational or interval (Arb) arithmetic, except where a README says
otherwise.
