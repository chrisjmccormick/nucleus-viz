<!-- md -->
# Where the Learning Happens: Per-Token Losses in RL vs. Fine-Tuning

<!-- md -->
In [part 1](link), we looked at how a reasoning model picks its next token, and found that it mostly... doesn't. Across 32,000 sampled responses to MATH-500 problems, **94%** of token positions were "singletons"--positions where the sampling nucleus contains exactly one token, and the model's choice is fully determined. The remaining few percent are the "branch tokens", and even those typically offer just 2-4 options.

In this post we'll look at what that structure means for training. When we reward a model for a correct response, or penalize a wrong one, *which tokens actually carry the lesson?*

Here's the answer, before any math. This is the same geometry/627 rollout from part 1, but now every token has a chip below it showing its training loss:

<!-- IMG: F1 — full-sequence loss strip, correct rollout (top portion).
     Source: html/math500_geometry_627_losses_correct.html -->
![PLACEHOLDER F1: correct rollout loss strip](TODO-F1)

A field of grey, with a handful of yellow flashes.

The grey chips are losses near zero--positions where training on this rollout teaches the model essentially nothing. The yellow chips are where all of the learning is concentrated, and if you compare them against the token colors above them, you'll see they're the branch tokens. Nearly the entire rollout is inert.

The rest of this post is about making that picture precise--and then breaking it, by showing what the same picture looks like for fine-tuning.

<!-- md -->
## 1 - The Advantage: One Number Per Rollout

Reasoning models are typically RL-trained with some variant of GRPO, and the algorithm descriptions can look intimidating--group sampling, reward normalization, clipped ratios. But from the perspective of an individual token, all of that machinery boils down to producing *one scalar*, called the **advantage**, which gets multiplied onto the token's loss.

Here's the actual calculation for our example. We sample a group of rollouts for the problem (our pool has 64), and score each one: reward 1 if the final answer is correct, 0 if not. Qwen2.5-Math-1.5B gets geometry/627 right in 15 of 64 attempts. Then each rollout's advantage is its reward, recentered and rescaled by the group statistics:

$$A = \frac{r - \text{mean}(r)}{\text{std}(r)}$$

With 15/64 correct, the mean reward is 0.234 and the std is 0.424, so:

- Every token of a **correct** rollout gets $A = +1.81$.
- Every token of an **incorrect** rollout gets $A = -0.55$.

That's it. The "group-relative" part of GRPO, the reward model, the normalization--they're all just different recipes for cooking up this multiplier. One number for the whole sequence, stamped identically onto every token.

There's a nice asymmetry hiding in those two values, by the way. Correct answers are the *minority* for this problem, so the formula automatically pushes harder on the rare successes (+1.81) than it punishes the common failures (-0.55). A problem the model usually gets right would flip that around.

<!-- md -->
## 2 - The Loss, One Token at a Time

With the advantage in hand, the per-token loss is just cross-entropy with a coefficient:

$$\text{loss}_t = -A \cdot \log p(\text{token}_t)$$

where $p$ is the probability the model assigns to the token it actually produced. Here are the first ten tokens of our correct rollout, with the arithmetic laid out:

<!-- IMG: F2 — opening 10 tokens, rows: token / −log p (nats) / × A / = loss.
     Source: html/math500_geometry_627_losses_opening_correct.html -->
![PLACEHOLDER F2: opening tokens loss arithmetic](TODO-F2)

Read any column top to bottom: the opening token "To" was a branch token--the model gave it probability 0.18--so $-\log(0.18) = 1.72$ nats, times the advantage of 1.81, gives a loss of 3.10. The advantage row is the same green +1.81 all the way across; that's not a rendering shortcut, that's the algorithm.

And look at what happens on the singletons. "problem" at $p = 0.99$ contributes 0.03. " to" at $p \approx 1.0$ contributes about **one thousandth** of a nat. For the model to learn from a token, there has to be a *gap* between what it predicted and what it did--and at a singleton, there is no gap. The loss formula doesn't need to be told about branch tokens; it finds them on its own.

> Two honesty footnotes. First, training computes these probabilities at $T = 1.0$ (the raw model), not the $T = 0.6$ we sampled with--so the chip probabilities here differ slightly from part 1's. Second, GRPO's actual objective wraps this in a PPO-style ratio with clipping, but on the first update step the ratio is exactly 1 and the gradient reduces to what's shown. The simple form is the real shape of the signal.

<!-- md -->
## 3 - Reading the Whole Rollout

Now the opening figure should make sense. Some things worth noticing as you scan it:

**The big losses are all "openers".** The ten highest-loss tokens in this 540-token rollout are 'First' (3.17), ' Now' (3.16), ' vectors' (3.14), ' understand' (3.11), 'To' (3.10), 'This', 'since'... These aren't the tokens doing arithmetic--they're the tokens choosing *how to proceed*. Sentence starts, clause pivots, the places where the response could have gone a different way. The math tokens--all those coordinates and equations--are singletons, locked in by the surrounding context.

**Even stopping is a singleton.** The final chip of the rollout is the end-of-sequence token at $p = 1.00$, loss 4e-04. The model is completely certain about when it's done, and reinforcing that certainty teaches it nothing.

**The grey isn't perfectly silent, though.** "Singleton" is a statement about the $T=0.6$ sampling distribution, but the loss is computed at $T=1.0$, where those same tokens aren't quite at probability 1.0. Scan the strip and you'll find sampling-singletons carrying small-but-real losses--"In" is a singleton at sampling temperature, but sits at $p = 0.84$ for the training policy, contributing 0.31. Hold that thought; it matters at the end of this post.

<!-- md -->
## 4 - The Other Sign: A Failed Rollout

What does the same picture look like when the model gets it wrong? Here's a rollout from the same group that reasons its way, quite confidently, to $\boxed{5}$ (the answer is 17):

<!-- IMG: F3 — opening 10 tokens of the incorrect rollout, red advantage chips.
     Source: html/math500_geometry_627_losses_opening_incorrect.html -->
![PLACEHOLDER F3: incorrect rollout opening](TODO-F3)

Identical machinery, one sign flip. The advantage is $-0.55$, so every loss goes negative--the gradient now pushes each of these tokens *down*.

And look at where the biggest push lands: the opening token "Let", at $p = 0.07$, takes a hit of $-1.43$--by far the largest in the rollout. RL isn't penalizing the model's arithmetic; most of the arithmetic was singletons here too. It's penalizing the *pathway*. "Let's solve the problem step-by-step" was an unusual way for this model to start, the attempt failed, and the lesson assigned is: that opening, in this context, is now a little less likely.

<!-- IMG: F4 — full-sequence loss strip, incorrect rollout (top portion).
     Source: html/math500_geometry_627_losses_incorrect.html -->
![PLACEHOLDER F4: incorrect rollout loss strip](TODO-F4)

Over many iterations, this is the whole game: branch tokens that lead to correct answers get pushed up, branch tokens that lead to failures get pushed down, and the model gradually re-balances the probabilities within each little nucleus. The singletons just come along for the ride.

<!-- md -->
## 5 - From Loss to Weights

A loss value is still an abstraction. What does one token's loss actually *do* to 1.5 billion parameters?

We can measure it. Take a single token, backprop only its loss, and record the gradient norm of every weight group in the model--the embedding table at the bottom, the 28 decoder layers, the LM head at the top. Here's that experiment for three tokens from the correct rollout: a hard singleton (" to", $p = 0.9994$), a branch token (" understand", $p = 0.18$), and a softer singleton (" the", $p = 0.94$):

<!-- IMG: F5 — graphviz gradient stacks, three columns, shared color scale.
     Source: figures/geometry_627_grad_layers.png -->
![PLACEHOLDER F5: per-layer gradient norms](TODO-F5)

The singleton column is flat grey, top to bottom--exactly as the loss chips promised. Its total gradient is roughly **1000x smaller** than the branch token's. Whatever your intuition says a "small" update is, one reinforced singleton is smaller: it is, for practical purposes, a no-op.

The branch token, meanwhile, reaches *everything*. The gradient is strongest at the LM head and the top layer, stays substantial through the middle of the network, and--this one surprised me--spikes again at the bottom, in layer 0 and the embedding table. A single branch token, one forward pass, and the entire depth of the model gets a meaningful nudge.

The middle column is the interesting one. " the" at $p = 0.94$ is a singleton by sampling standards, but its gradient is only ~15x smaller than the branch token's--and ~70x *larger* than the hard singleton's. Singletons are not a monolith: the nearly-certain ones are true no-ops, but the merely-confident ones still carry real signal. (This is exactly the regime that DAPO-FT pokes at--they find that training on branch tokens alone outperforms training on everything, and that training on *only* the singletons is actively harmful. The gradient says the singletons aren't all zeros, and apparently what's in them isn't all good.)

One more detail from this experiment, because it completes the mechanism from section 4. Looking at *which rows* of the LM head receive gradient for the branch token " understand": the target row gets the largest share, as you'd expect. But the second-largest goes to " use"--which happens to be the model's actual favorite at that position ($p = 0.64$)--and its gradient points the *other way*. A GRPO step at a branch token is mostly a **transfer of probability between nucleus members**: push the sampled-and-rewarded option up, push its main competitor down. The vocabulary's other 151,930 rows receive crumbs in proportion to their (tiny) probabilities. RL doesn't write new options into the nucleus; it re-weights the menu that's already there.

<!-- md -->
## 6 - Fine-Tuning Is a Different Animal

Everything above has a property that's easy to miss because it's so structural: **RL trains on text the model produced itself.** Every training token was, by construction, sampled from the model's own nucleus. The branch/singleton geometry of part 1 isn't just describing the rollouts--it's describing the entire universe of text RL can ever train on.

Fine-tuning has no such constraint. The training data can come from anywhere, and can disagree with the model completely.

To see what that looks like, I took the MATH dataset's official reference solution for our problem and teacher-forced it through the model as if it were the model's own response, computing the same per-token probabilities and losses (plain cross-entropy now--no advantage, this is SFT):

<!-- IMG: F6 — full SFT strip, dataset solution, burgundy OOD chips + extended ramp.
     Source: html/math500_geometry_627_losses_solution.html -->
![PLACEHOLDER F6: dataset solution SFT strip](TODO-F6)

It's a different world. The stats flip: on its own rollout, the model was 86% singletons; on the textbook's solution, only 48%--with **11% of the tokens falling outside the model's nucleus entirely** (the burgundy chips). Those are tokens the model considers effectively impossible. The loss values are on a different scale too: the largest per-token loss in the RL figures was 3.17; here the peak is **10.5**, and I had to extend the color ramp into orange just to display the range.

The very first token sets the tone. The reference solution opens with "Name the points..."--and "Name" gets $p = 0.0008$. Recall from part 1 that this model's opening nucleus for this problem is {The, To, Please, Let}. "Name" was never on the menu. Cross-entropy slams it with 7.1 nats at token one.

Even the *ending* is out of distribution: the EOS token after the solution's final sentence gets $p = 0.0009$, loss 7.05. Stopping where the textbook stops is, itself, something this model would never do.

<!-- md -->
### The model reads the answer and still gets it wrong

My favorite detail in this entire experiment is hiding near the end of the strip.

<!-- IMG: F6-crop — the "(8, 9)" region of the solution strip (steps ~133-160).
     Source: html/math500_geometry_627_losses_solution.html -->
![PLACEHOLDER F6-crop: the (8,9) region](TODO-F6-crop)

By this point in the solution, all the reasoning is done. The text has established that $D$ is "two units to the right and one unit up from $B$", with $B = (6, 8)$ sitting right there in context. Then it writes the answer: "the coordinates of $D$ are $(8, 9)$".

The "8" is burgundy. With the entire correct rationale force-fed into its context, the model gives the correct x-coordinate a probability of 0.09--outside its nucleus. And it's worse than uncertain: the model's nucleus at that position is a *singleton*, and the token in it is "9". It wanted to lead with the y-coordinate. Two tokens later, given the "(8," for free, the model offers the correct "9" at only a 10% probability--its preferred digit there is "5".

Reading the right reasoning is not the same as being able to produce it. The model can follow the textbook all the way to the finish line and still trip over the tape--twice.

And this is exactly the point where the RL and FT pictures snap together. Under nucleus sampling, a token outside the nucleus doesn't have a *small* probability of being generated--it has **zero**. The model will never write that "8" on its own, which means no rollout will ever contain it, which means no reward signal can ever reach it. On-policy RL is structurally incapable of teaching this lesson. Teacher forcing is the only mechanism that can put loss on that token--and when it does, the loss is enormous: the biggest per-token losses here run ~3x the largest ones RL produces on this same problem.

I ran the gradient experiment on these tokens too, expecting the gradients to also come out 3x bigger. They don't--and the reason is a property of cross-entropy worth knowing: **its gradient saturates even though its loss doesn't.** The push on the target's logit is proportional to $(p - 1)$, which is bounded--at $p = 3\text{e-}5$ it's $\approx -1.0$, barely harder than at $p = 0.18$. So " first" carries 3.4x the loss of our branch token " understand", but a slightly *smaller* gradient. Past a point, being more wrong doesn't push harder; cross-entropy's push maxes out at "wrong," and the thing that actually scales RL's updates beyond it is the advantage multiplier.

<!-- IMG: F7 — graphviz gradient stacks for the SFT tokens (Name / first / 8 / 7).
     Source: figures/geometry_627_grad_layers_sft.png -->
![PLACEHOLDER F7: SFT per-layer gradient norms](TODO-F7)

What the FT update *does* have is perfect aim. At the "8", the two largest gradients in the LM head are the "8" row, pushed up--and the "9" row, the model's wrong singleton ($p = 0.81$), pushed down. One teacher-forced token simultaneously installs the right digit and dismantles the confident wrong one. And the singleton control behaves exactly like its RL counterpart: the "7" at $p = 0.999$ produces the same near-zero gradient under SFT as " to" did under RL. Singletons are no-ops in both training modes; the difference between RL and FT is entirely about which *non*-singletons each one is allowed to touch.

The two training modes aren't competing tools; they have different jobs:

- **RL re-balances the tree the model already has.** Its signal lives at the branch tokens, and its updates shift probability between options that were already in the nucleus.
- **FT carves new paths into the tree.** Its largest signals are precisely the tokens the model would never produce--and it bulldozes them in whether the surrounding distribution likes it or not.

<!-- md -->
## Takeaways

1. The advantage--for all of GRPO's machinery--is a single scalar stamped on every token of a rollout. It sets the sign and size of the lesson; *where* the lesson lands is decided entirely by the per-token probabilities.
2. RL's learning signal concentrates almost entirely at branch tokens. Per-token losses find them automatically, the gradients confirm it (a branch token moves the whole model; a hard singleton moves ~nothing, ~1000x less), and at each branch the update is mostly a probability transfer between nucleus members.
3. "Almost nothing" is not zero--softer singletons carry real gradient, which is why how you treat them (mask them? train them? DAPO-FT) is an actual design decision.
4. RL can only re-weight pathways the model can already reach. Text outside the nucleus has zero sampling probability, so fine-tuning is the only way to put loss there--and that's both its power and its bluntness.

In part 3, we'll get to the question this all sets up: if RL's job is re-balancing the branch tokens, which branches *should* it re-balance? It turns out that for this model, on this benchmark, you can predict a shocking amount about a rollout's success from a single branch token--the first one.
