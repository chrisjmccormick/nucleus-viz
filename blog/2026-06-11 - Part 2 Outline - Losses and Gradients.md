# Part 2 Outline — Where the Learning Happens: Per-Token Losses in RL vs. Fine-Tuning

> Working outline. Each section lists: the goal, the points to make (with the real
> numbers from our figures), and the visual + placement. Depth-vs-gradient topic
> deliberately held for later.
>
> Figure assets (HTML in `html/`, render to PNG for the post like part 1):
>   F1  math500_geometry_627_losses_correct.html        (full strip, A = +1.81)
>   F2  math500_geometry_627_losses_opening_correct.html (token / −log p / ×A / =loss)
>   F3  math500_geometry_627_losses_opening_incorrect.html
>   F4  math500_geometry_627_losses_incorrect.html      (full strip, A = −0.55)
>   F5  figures/geometry_627_grad_layers.png            (graphviz stacks)
>   F6  math500_geometry_627_losses_solution.html       (SFT, burgundy OOD chips)
>   F7  figures/geometry_627_grad_layers_sft.png        (SFT gradient stacks)
>   F8  figures/geometry_627/Grad-Rows-01-rl-branch.png  (head-row gradients, ring layout;
>       -02-rl-negative and -03-sft-ood are the A<0 and SFT companions — candidates for
>       sections 4/5's "which rows" discussion)

---

## 0. Hook + recap (short)

- One-paragraph recap of part 1: at T=0.6/top-p 0.95, ~94% of tokens are
  singletons; the branch tokens are few and small. Question for this post:
  *when we train on a rollout, which tokens actually teach the model anything?*
- **Visual (the hook): F1**, placed immediately — the same geometry/627 rollout
  from part 1, now with a loss chip under every token. The reader sees it before
  the math: a field of grey with a handful of yellow flashes. "Almost the entire
  rollout contributes nothing."
- Note the page header elements (problem card, stats row: 540 tokens / 462
  singletons (86%) / 78 branch (14%) / ✓ correct) — they carry part 1's stats
  forward without re-explaining.

## 1. The loss, one token at a time

- Goal: make `loss_t = −A · log p(token_t)` feel obvious, then make "A is just a
  scalar" the simplifying insight.
- Points:
  - RL trains on the model's own rollouts. The reward is assigned to the *whole
    sequence*; per-token, all that arrives is one number: the advantage.
  - Demystify GRPO/DAPO here (in text, not the figure): reward 1/0 per rollout,
    A = (r − mean)/std over the group. Our group: 64 rollouts, 15 correct →
    A = **+1.81** for a correct rollout, **−0.55** for an incorrect one. All the
    group-relative machinery is "various ways of coming up with a scalar
    multiplier on the loss."
  - Nice aside: correct answers are the minority (15/64), so the rare successes
    get the *larger* advantage magnitude — the formula automatically pushes
    harder on rare wins.
  - The per-token loss is then just cross-entropy scaled by A. Walk one column:
    'To' = 1.72 nats × 1.81 = 3.10.
  - Footnote-level honesty: training uses T=1.0 (no sampling temperature) and
    the PPO-style ratio is 1 at the first update step, so clipping doesn't
    enter; this is the plain policy-gradient form.
- **Visual: F2** (opening 10 tokens; rows token / −log p / × A / = loss). The
  green advantage chips visibly identical on every token = the point.

## 2. Reading the whole rollout through the loss lens

- Goal: connect loss values back to the branch/singleton split from part 1.
- Points (all from F1):
  - Branch tokens carry essentially all of the loss. The top-10 loss tokens are
    all "discourse openers": 'First' (3.17), ' Now' (3.16), ' vectors' (3.14),
    ' understand' (3.11), 'To' (3.10), 'This', 'since' ... — the places where
    the model chose how to *proceed*, not what word completes a phrase.
  - Singletons ≈ zero loss. The trailing EOS is itself a singleton (p = 1.00,
    loss ≈ 4e-04): the model is certain when to stop, and reinforcing that
    teaches nothing.
  - Subtlety worth a paragraph (sets up DAPO-FT later): "singleton" is a T=0.6
    sampling statement, but the training loss is computed at T=1.0 — where those
    same tokens aren't exactly p = 1.0. Some grey tokens carry visible loss
    (e.g. 'In': sampling-singleton, but p = 0.84 at T=1.0 → loss 0.31).
    Singleton ≠ mathematically zero signal, just small.
- **Visual: F1** again (reader is now equipped to read it); can crop a row or
  two as inline callouts (e.g. the 'understand' row, the EOS chip).

## 3. The other sign: a failed rollout

- Goal: same machinery, negative coefficient.
- Points:
  - Incorrect rollout (sample 36 — a clean wrong attempt that reasons to
    \boxed{5}; answer is 17). A = −0.55: every per-token loss flips sign; the
    gradient now pushes the chosen tokens *down*.
  - The largest-magnitude token is the opener 'Let' (p = 0.07, loss = −1.43):
    the rollout's most "characteristic" choice takes the biggest hit — pushing
    down the pathway, not the arithmetic.
  - Reiterate asymmetry: |−0.55| < |+1.81|; failures are common in this group,
    so each one is individually a weaker signal.
- **Visuals: F3** (opening rows, red advantage chips) then **F4** (full strip).
  Could show F3 alone if the post is running long; F4 is the same lesson at
  scale.

## 4. From loss to weights: what one token does to the model

- Goal: losses are numbers; show the gradient they actually produce in 1.5B
  parameters. This is the "gradient magnitudes" half of the post.
- Points:
  - Setup in one sentence: backprop a *single token's* loss and measure the
    gradient norm of every weight group (28 decoder layers + the embedding
    table's two roles, separated analytically since Qwen ties them).
  - Singleton 'to' (p = 0.9994, loss 1e-3): flat grey, top to bottom. ~1000×
    smaller than the branch token — one near-certain token is a true no-op.
  - Branch 'understand' (p = 0.18, loss 3.07): the entire model lights up —
    head 142, top layer 49, a working middle, and (surprise) a spike at the
    bottom (layer 0: 123, embeddings: 98). Gradient is largest at both ends.
  - The middle case 'the' (p = 0.94, loss 0.10): ~15× below the branch but
    ~70× above the hard singleton. **Singletons are not a monolith** — this is
    the empirical hook for DAPO-FT (training only branch tokens helps;
    training only singletons is harmful — they do carry *some* signal).
  - LM-head rows: not just the target. For 'understand', the target row gets
    norm 112 — and ' use' (the model's actual T=1.0 favorite at p = 0.64) gets
    87, pushed the *other way*. A GRPO step at a branch is mostly a probability
    *transfer between nucleus members* ('use' → 'understand'). This is the
    mechanism behind part 1's claim that RL "re-distributes probabilities
    within branch nuclei."
- **Visual: F5** (three graphviz stacks, shared color scale). Optional small
  table of the top-5 head rows for 'understand'.

## 5. Fine-tuning is a different animal

- Goal: the RL picture above is *on-policy*: every training token came from the
  model's own nucleus tree. SFT removes that constraint — show what happens.
- Points:
  - Setup: teacher-force the dataset's reference solution as the response;
    loss is plain cross-entropy (no advantage — or A ≡ 1).
  - The stats flip: only 48% singletons (vs 86% on its own rollout), and
    **17 of 161 tokens (11%) are outside the model's nucleus entirely**
    (burgundy). Max per-token loss 10.46 vs 3.17 on the rollout — the loss
    ramp needs a deeper color scale just to display it.
  - Tour the burgundy: the opener 'Name' (p = 8e-4 — never in part 1's
    {The, To, Please, Let} nucleus → 7.1 nats on token one); ' first'
    (p = 3e-5 → 10.5 nats); and the closer: even the **EOS is
    out-of-distribution** (p = 9e-4, loss 7.05) — stopping where the textbook
    stops is itself something the model would never do.
  - **The (8, 9) story** (own subsection; it's the best moment in the post):
    at "(", the gold text needs '8' — the model's nucleus there is a
    *singleton*, and it's '9' (96% at T=0.6). It wanted to lead with the
    y-coordinate. Two tokens later, given "(8," for free, the correct '9' is a
    10% pick (it prefers '5'). Force-fed the entire correct rationale, the
    model still garbles the answer two ways. Reading the right reasoning ≠
    being able to produce it.
  - The RL-vs-FT punchline: a token outside the nucleus has *literally zero*
    probability under top-p sampling — not small, zero. On-policy RL can never
    visit '8' along this pathway, so no reward can ever reinforce it. Teacher
    forcing is the only mechanism that can put loss there. The same property
    that makes singletons useless to RL ("nothing to learn where the model is
    certain") is the failure mode FT exists to fix ("certain of the wrong
    thing, reachable only by off-policy data").
  - Corollary (one line): FT's gradients on OOD tokens are ~3× the largest RL
    gradients on the same problem — FT doesn't rebalance the tree, it
    bulldozes new paths through it.
- **Visual: F6** (full SFT strip; burgundy chips + hot ramp + the
  outside-nucleus stat block). Inline crop of the "(8, 9)" region for the
  subsection.

## 6. Takeaways + part 3 tease

- The three-line summary:
  1. RL's learning signal lives almost entirely at branch tokens; the advantage
     is just a scalar that sets sign and size.
  2. A single branch token moves the whole model; a single singleton moves
     ~nothing — but "almost nothing" is not zero, and the borderline singletons
     are why singleton-handling (DAPO-FT) matters.
  3. RL rebalances pathways the model can already reach; FT injects pathways it
     can't. Different tools, visible per-token.
- Tease part 3: if RL's job is rebalancing branch tokens, which branches *should*
  it rebalance? → the oracle / branch-prediction experiment ("One Token To Rule
  Them All" notes from the part-1 draft).

---

## Production notes

- All six figures regenerate from caches: `python html_token_losses.py` (F1–F4,
  F6) and `python grad_layers.py --render-steps 7 8 9` (F5). PNG-ify the HTML
  strips the same way as part 1's figure.
- Numbers quoted above are all real/verified: group 15/64; A = +1.807/−0.553;
  losses from the cached traces; gradient norms from the Colab bf16 run.
- Held for a later post: gradient magnitude vs. token depth (the
  `..._depth.png` figure and the position-effect analysis).
- The blog text owns: the GRPO/DAPO advantage formula, the T=1.0-vs-T=0.6
  footnote, the PPO-ratio-equals-1 footnote (figures deliberately show none of
  this).
