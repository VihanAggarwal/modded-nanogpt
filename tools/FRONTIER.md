# The track-1 frontier (open PRs on top of record #92, surveyed 2026-10-08)

Upstream master is still `4ea6b93` (ANVIL2, 39.9 s; the `track_1_short` refactor). Every open PR with a claimed
track-1 gain was read, its logs re-parsed where it has any, and each verdict checked by a second reviewer.
"Gain on this fork" is the expected time saved if added on top of master + our systems patches + #375 + #379.

| PR | change | evidence (re-parsed) | state | gain on this fork | verdict |
|---|---|---|---|---|---|
| #379 (NathanGodey, yoavartzi) | CPLM copy-sink pointer mixture | n=8, val 3.27686, p=5.1e-4; 36.009 vs same-node master 40.575 s | open | the base gain, -4.2 to -4.6 s | **in the fork** |
| #375 (Daniel Monroe) | token-normalized n-gram hashes | n=7+7 (description only, no logs): -2.0 mnat and -0.42 s at -15 steps | open | 2-6 mnat of margin at 978 steps; ~-0.45 s if 963 holds | **in the fork** |
| #371 (romeerpillay) | sampled LM-head dW + truncated attention backward | n=12, p=0.0024, -0.76 s vs #360 | open | ~0.3-0.5 s, needs a fix for CPLM's `<copy>` gradient and its own pool | not integrated |
| #382 (krishaab) | stack of #371 head half, #375, #376 + systems pieces | n=18, val 3.27925 (3 of 18 > 3.28), -3.6 s vs master on Modal | open | ~0.3-0.8 s (most of the headline is a step cut or overlaps CPLM) | not integrated |
| #376 (cghane) | document-local copy mixture at validation | logs on 1xH100 | open | ~0 (CPLM covers the same long-range copies) | skip |
| #366 (Nihir Patel) | n-gram rows from host RAM | pre-ANVIL2 | open | about -1 s net loss on ANVIL2 | skip |
| #372 (Lev Berman) | adaptive softmax | closed by its author: sampled softmax beats it | closed | ~0 | skip |
| #358, #364 | norm CSE, fp8 up-proj; endgame EMA | already in ANVIL2 in better form | closed | 0 | skip |
| #367, #380, #381 | exact-match retrieval and counts over all 103 train shards | 21.6 s, 9.65 s, (#381 unvalidated) | open | if ruled in, the record falls to ~9.65 s | needs a maintainer ruling on train-shard and CPU use |

How records are judged (from the merged PRs #317-#360): p = `scipy.stats.ttest_1samp(vals, 3.28,
alternative='less').pvalue` over all counted runs; a same-node baseline of the merged record; the maintainers re-time on
their own nodes and credit each incorporated PR only its own increment, so a stack also measures itself against the open
PR it builds on. No PR has been merged without the author's own H100 runs. `tools/record_attempt/` runs exactly that
protocol in one command.
