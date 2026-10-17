# Licence of the files in this folder

`sample_corpus.json`, `bm25_index.json` and `examples.json` contain text from Wikipedia articles as distributed in
SQuAD 2.0 (Rajpurkar, Jia and Liang, 2018, <https://rajpurkar.github.io/SQuAD-explorer/>), which is licensed
**CC BY-SA 4.0** (<https://creativecommons.org/licenses/by-sa/4.0/>).

- The passages are shown unmodified. `sample_corpus.json` cuts them into overlapping chunks of words, and
  `bm25_index.json` is a token index of those chunks.
- `examples.json` also holds questions from the SQuAD 2.0 dev set, the answers produced by the evaluated reader and the
  outcome category recorded by the committed evaluation run.
- These files are licensed CC BY-SA 4.0, like the rest of this repository. Reuse them with attribution to
  Rajpurkar, Jia and Liang (2018) and to the Wikipedia authors of the original text.

See `THIRD_PARTY_NOTICES.md` in the repository root for the licence evidence.
