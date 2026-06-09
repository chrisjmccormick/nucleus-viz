<!-- md -->
# ▂▂▂▂▂▂▂▂▂▂▂▂

<!-- md -->
# Branch Tokens in Reasoning Traces

<!-- md -->
Reasoning models are surprisingly deterministic in the responses they generate. 

The below illustration shows a response sampled from Qwen2.5-Math-1.5B a challenging geometry problem:

> "The coordinates of a parallelogram are (5, 3), (6, 8), (7, 4) and $(x, y)$ and $x > 7$. What is the value of $x + y$?"

In each row, the top line of tokens is the generated response (beginning with "To solve this problem, we need to understand the properties of a parallelog"). For positions where we had multiple tokens we could sample from, the alternatives are listed, and each token shows its probability of being chosen. 

![Sampled response to the geometry problem](https://raw.githubusercontent.com/chrisjmccormick/nucleus-viz/main/html/math500_geometry_627_to_solve.png)

> You can view the full 540 token response [here](link) to see how the model solves this problem and arrives at the correct answer of 17.

Two things immediately stand out:
1. At most postitions, we are sampling from a pool of 1--the prediction is deterministic.
2. Where there are options, there are surprisingly few. 

We'll refer to the sampling pool at each position as its "nucleus"; this term comes from top-p "nucleus sampling", which filters down the SoftMax probabilities of the entire vocabulary down to just those that capture 95% of the total (`top-p = 0.95`).

Positions with only a single possible prediction are called "singletons", and we'll refer to the non-singletons as "branch tokens".

The statistics on singletons vs. branches are remarkable. Over a collection of xxx (64 x 500) responses to Math500 benchmark problems (xx million tokens in total):
- **94%** of the tokens are singletons.
- The branch nuclei are small--their average size is only **xxx**.
    - TODO - This stat was intended to exclude singletons; redo it.
- The branches are concentrated towards the beginning of the response. 

In this series of posts, we'll review how sampling works and look deeper into these nuclei, then see how this determinism affects the per-token losses during Reinforcement Learning, by looking at the math of per-token losses, and finally look at how recent RL techniques exploit this "low entropy" quality of reasoning models.

Let's start by reviewing the core concepts of token prediction.

## 1 - Calculating Token Probabilities

At each position, we use the output of Qwen2.5-Math-1.5B to pick a next token from its 151,936 token vocabulary. 

The process breaks down into:
1. Calculating the output "logits".
2. Applying the "Temperature" coefficient to each.
3. Normalizing the logits into a probability distribution via SoftMax.
4. Filtering out the low probability choices, retaining just the top-p.
5. Re-normalizing within the nucleus and choosing a random token, weighted by probability. ("Sampling from the distribution")

We'll illustrate this process on the opening branch token (with choices "To", "The", "Please", "Let") of Qwen's response to this geometry problem, starting with the logits.

### 1.1 Logits

Token prediction occurs one at a time. To choose the next token, we feed the most recent one into the model, and the output of the final Transformer layer is a single vector. The next step is the largest matrix multiplication in the entire model--we dot product the 1,536 dim output vector with its 1,536 x 151,936 language modeling head (467 million flops) to produce 151,936 raw logits.

We can visualize these logits by wrapping them into a square and plotting their heat map. It's a noisy picture when displayed in vocabulary order, so we've laid them out in concentric square rings: the highest-scoring token sits in the center, and tokens expand outward, ordered by the size of their logit. 

![Raw logits across the full vocabulary](https://raw.githubusercontent.com/chrisjmccormick/nucleus-viz/main/figures/geometry_627/Vocab-01-Raw_Logits.png)

The result is a kind of pyramid shape, with a gradual slope away from the center. This matches my normal intuition of the prediction--there are some high probability top choices, and the confidence gradually drops over the rest of the vocabulary. 

Yet, that intuition is actually very wrong, and the reason is the SoftMax. 

Before applying the SoftMax, we apply "Temperature" by dividing every logit by `T = 0.6`. The effect of temperature is easier to understand after we've first we've gained some intuition about the SoftMax, so we'll go there next.

## 3. Temperature: widening (or narrowing) the gaps

- The next step in sampling is to apply a coefficient to the logits which ultimately sharpens the distribution.
- This coefficient is called the **temperature** `T`.
- Interestingly, a temperature of `1.0` is very (permissive?) For reasoning tasks in particular, we'll use a lower temperature, `0.6` here.

- Greedy decoding, where we simply take the token with the highest logit (argmax of the logits) can also be expressed by setting the temperature to ~0.0, (which turns the value into infinity)

- We'll see in the next section that what matters for SoftMax are the deltas--the gaps between the logit values. 

Dividing by a number smaller than 1 makes everything bigger — and, crucially,
it stretches the **gaps**. Watch three logits at `T = 0.6`:

| logit | ÷ 0.6 |
|------:|------:|
| 15    | 25.00 |
| 10    | 16.67 |
| 5     | 8.33  |

The values 15 and 10 were 5 apart; after dividing by 0.6 they're 8.33 apart.
Every gap is multiplied by `1 / 0.6 ≈ 1.67`. Temperatures **below 1 sharpen**
the distribution (wider gaps, more decisive); temperatures **above 1 flatten**
it; `T = 1` changes nothing.

![Logits after temperature scaling](https://raw.githubusercontent.com/chrisjmccormick/nucleus-viz/main/figures/geometry_627/Vocab-02-Temp0.6.png)

(TODO - It's not invisible.)

On the same linear color scale this looks almost identical to the raw logits —
dividing by a constant is just a rescale. The effect is invisible *here*. But
remember the previous section: softmax is about to exponentiate these gaps, and a
1.67× stretch in the gaps becomes an enormous swing in the final probabilities.

If you stopped here, you might guess the model is fairly undecided — lots of tokens look "close."

That intuition is wrong, and the reason is what we do with these scores next.

## 2. Softmax: turning scores into probabilities

Logits aren't probabilities. They can be positive or negative, and they don't
add up to anything in particular. **Softmax** fixes that: it exponentiates every
score and divides by the total.

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Exponentiating makes every value positive, and dividing by the sum makes them
add to 1 — a valid probability distribution. But the exponential does something
more important than bookkeeping: it turns **additive** gaps between scores into
**multiplicative** ones.

Here's the whole trick in three lines. The constant `e ≈ 2.72`, and every time
the exponent goes up by 1, you multiply by another `e`:

```
e^1 = 2.72
e^2 = 2.72 × 2.72
e^3 = 2.72 × 2.72 × 2.72
```

TODO: List the top four token's logits and their exp(logit) value


So a token whose logit is just **1 point** higher than another isn't slightly
more likely — it's about **2.72×** more likely. Two points higher:
`2.72 × 2.72 ≈ 7.4×`. Ten points higher: over **22,000×**. A gently sloping
logit landscape, pushed through `e^x`, turns into a spike.

One more property worth noting: softmax only cares about the **differences**
between logits, not their absolute size. Adding the same constant to every logit
cancels out in the numerator and denominator. All that matters is how far each
token sits below the leader. 
(TODO - More interesting point--you could shift the logits to be all positive, 0 - 30 instead of -15 to 15, and the softmax would still be the same.)

## 4. Put them together: the probability collapses

Apply softmax to the temperature-scaled logits — `softmax(logits / 0.6)` — and
that smooth landscape from step 1 collapses. Almost the entire vocabulary rounds
to a probability of zero, and a single token lights up dead center:

![Softmax probability collapses onto a few tokens](https://raw.githubusercontent.com/chrisjmccormick/nucleus-viz/main/figures/geometry_627/Vocab-03-Softmax.png)

(In this plot anything below ~0.004 is floored to pure black, so the black sea
really is "essentially zero probability.")

Here are the top tokens. The `exp` column is `e^((zᵢ − z_max)/T)` — the
exponential measured *relative to the top token*. We subtract the max first
because the absolute exponentials are huge and only ratios matter (this is the
standard log-sum-exp trick). Divide any `exp` value by the total and you get the
probability.

| rank | token | logit | logit / 0.6 | exp (rel. to top) | probability | cumulative |
|-----:|:------|------:|------------:|------------------:|------------:|-----------:|
| 1 | `The`    | 14.19 | 23.65 | 1.0000 | 0.5809 | 0.5809 |
| 2 | `To`     | 13.69 | 22.81 | 0.4346 | 0.2525 | 0.8334 |
| 3 | `Please` | 12.81 | 21.35 | 0.1011 | 0.0587 | 0.8921 |
| 4 | `Let`    | 12.81 | 21.35 | 0.1011 | 0.0587 | **0.9509** |
| — | — | — | — | — | — | **← top-p = 0.95 cuts here** |
| 5 | `Given`  | 12.06 | 20.10 | 0.0290 | 0.0168 | 0.9677 |
| 6 | `If`     | 11.13 | 18.54 | 0.0061 | 0.0035 | 0.9712 |
| 7 | `What`   | 11.13 | 18.54 | 0.0061 | 0.0035 | 0.9747 |
| 8 | `This`   | 11.06 | 18.44 | 0.0055 | 0.0032 | 0.9779 |
| 9 | `\\`     | 11.00 | 18.33 | 0.0049 | 0.0029 | 0.9808 |
| 10 | `You`   | 11.00 | 18.33 | 0.0049 | 0.0029 | 0.9836 |

**Sum of `exp (rel. to top)` over all 151,936 tokens = 1.7214.** That single
number is the entire denominator of the softmax. Notice it's barely above 1.0 —
the top token alone contributes `1.0` of it, which is exactly why its
probability is `1.0 / 1.7214 = 0.5809`. The other 151,935 tokens *combined* add
only 0.72. The drop-off is brutal: by rank 6 the exponential has fallen by more
than 99% from the top, even though the logit only slipped from 14.19 to 11.13 —
a gap of just 3 points, turned into a ~165× difference by `e^(3/0.6)`.

## 5. The nucleus: four tokens out of 151,936

This is what makes **nucleus (top-p) sampling** work. Instead of considering all
151,936 tokens, top-p sampling keeps only the smallest set whose probabilities
add up to `p` (here 0.95) and samples from those.

Zooming into the bright center — the top 49 tokens, each labeled with its actual
softmax probability — shows how quickly the mass runs out:

![The few tokens carrying real probability](https://raw.githubusercontent.com/chrisjmccormick/nucleus-viz/main/figures/geometry_627/Vocab-04-Softmax-Nucleus.png)

The token in the dead center is **`The`**, at probability **0.5809** — by itself
more than half of everything. Ringed around it are the only other tokens with
any real weight.

Apply the top-p = 0.95 cutoff and almost all of them vanish. The cumulative
probability crosses 0.95 the instant `Let` is added (0.9509), so the nucleus is
exactly **four tokens** — `The`, `To`, `Please`, `Let`. `Given` and everything
else fall outside it and drop to black.

Sampling then **renormalizes within the nucleus** — it divides each surviving
probability by their sum (0.9509) so the four add back up to 1. That's the
distribution the model actually draws its first word from:

![The top-p = 0.95 nucleus](https://raw.githubusercontent.com/chrisjmccormick/nucleus-viz/main/figures/geometry_627/Vocab-05-Top-p0.95-Nucleus.png)

| token | softmax prob | renormalized (÷ 0.9509) |
|:------|-------------:|------------------------:|
| `The`    | 0.5809 | 0.6110 |
| `To`     | 0.2525 | 0.2655 |
| `Please` | 0.0587 | 0.0618 |
| `Let`    | 0.0587 | 0.0618 |

`The` ends up carrying **61%** of the nucleus, still parked in the center.

## Takeaway

The model scores 151,936 tokens on a smooth, almost continuous-looking scale.
But softmax runs those scores through `e^x`, which converts gentle additive
differences into violent multiplicative ones — and temperature `T < 1` widens
the gaps before they're exponentiated, making the spike sharper still. The result
is that essentially all of the probability piles onto a handful of tokens. After
a top-p = 0.95 filter the model's first word is really a choice among just
**four**, led by `The` at the center with 61% of the renormalized mass.

That extreme concentration is exactly what nucleus sampling is built to exploit:
most of the vocabulary was never in the running to begin with.







---

In this series of posts, we'll visualize the next token prediction choices of reasoning models, review how "sampling" works, 
- Language models predict the next token by computing logits over their entire vocabulary--i.e., for every word, there's a calculated value that reflects the model's confidence in that choice.
- The SoftMax function turns those logits into probabilities, and we pick the next token "at random", except weighted by those probabilities. 
- When you look at the actual probabilities, however, particularly in reasoning oriented tasks, it's surprisingly more deterministic.
- The token probabilities don't gently slope away towards less-and-less likely choices. Instead the model typically defines a sharp cutoff--a probability _cliff_. 
- Additionally, when we pick the next token, we don't actually consider the full vocabulary--we use a heuristic to try and detect that cliff called "nucleus sampling". 
- It places a hard cutoff on which tokens we actually consider, and it's defined by setting the "top-p" value. For reasoning tasks, a standard top-p value is 0.95, which means that we accumulate the most probable tokens until we've reached a total probability of 0.95. 
* There are a couple things that are quite striking about these token "nuclei".




geometry/617

Number Theory 515
"What is the smallest positive perfect cube that can be written as the sum of three consecutive integers?"

Answer: 27

(Because $3^3 = 27$ and $8 + 9 + 10 = 27$)

The full response to this question has xx tokens, of which xx (xx.x%) have singleton nuclei.
Of the branch tokens, the min, median, max of their nuclei are x / x / x.

> Same AI agent that helps you with the visualization, do this one.

Here is the response from Qwen 2.5 Math 1.5B--a smaller model which has been trained for 'reasoning' about math, just not with explicit < think > tokens.

> Hopefully it's interesting on both fronts
> It'll be the one we use further down for branch - oracle, so pick based on that?

**Per-Token Losses**
- For the model to learn, their must be a gap in probability. 
	- Large gaps produce large gradients, small gaps produce none
	- For example, if the model already produces a singleton, and we reinforce that choice, there's no gradient.
- Because RL only trains on responses it produced itself, the training data can only come from this tree of possible responses. 
- Singletons produce no gradient.
- So the only place where there is mathematically any significant learning signal is at these branch tokens.
- (Note - With teacher forcing / SFT, the data can come from anywhere, and can completely disagree with the model)

**GRPO**

Reasoning is typically trained via GRPO, which alters the loss by multiplying it with a positive or negative value based on the reward assigned to the response.

Notably, the rewards are assigned based on the whole sequence, so every token has the same "Advantage" multiplier applied to it.

Since GRPO has more-or-less no effect on the singletons, it boils down to this:

By construction, the training token, the target, has to come from this nucleus. 

Let's say it's the word "To". Qwen predicts "To" with a very low probability of xxx, so the loss will be:

(math here)

The rollout was correct, so the loss will be

(math)

Very high.

Over time, the model will learn to re-distribute the probabilities within these branch token nuclei to close down lower performing pathways. 

Todo: Most branches seem to retain their nuclei, and just have it rebalanced. It is possible, though, for the model to gradually fully push a token out of the nucleus.

Note: It's also possible for the model to bring a new one in! We'll look at that later.

**Branch Prediction**

There's a simple, albeit rather expensive, way to see what GRPO will do to a branch token over many iterations.

This works best on problems that the model struggles on, where a good portion of its responses are correct, but the rest are not.

We can pick any branch token in a rollout, take each of the tokens in its nucleus, and generate a large number of responses along each pathway to measure the probability of success of each pathway.

**One Token To Rule them All**

Running this experiment using the small Qwen 2.5 Math 1.5B model on the Math500 benchmark has some pretty amusing results. 

For this problem, the first token nucleus has 4 options in it. If we generate 16 responses along each of these and score them, we can estimate the accuracy along each branch, and compare it to what the model currently predicts.

For this model, on this benchmark, the choice of opening token has incredible predictive power for the accuracy of the model's response.

If you do this for all 500 questions in the benchmark, here's what you'll find.
- Sampled: Sample from the model's probability distribution. i.e., standard decoding.
- Uniform: Ignore the model's probabilities and pick a starting token uniformly at random.
- Oracle: Pick the best starting token (after trying all of them a number of times)
- Oat-Zero: The Qwen model tuned via GRPO on the training set for this benchmark

(Table)

The table's showing that:
1. Uniform outperforming means don't listen to this model's advice on the first token. It's somehow calibrated _against_ the correct answer.
2. If you figure out the right starting token, you'll get most of the way to a benchmark score.

If we only ever trained this one token (masking off the rest of the response) we can intuit what will happen to its nucleus over time:
- If it's high probability but underperforming, it will be visited often and gradually shoved out of the nucleus. 
- If two tokens have a equal probability, and both lead in good directions, but one is better, it's going to exploit the better pathway, and shut down the second best.
- 
Underperforming pathways will be shut down and successful ones will be made more likely. 

We can also intuit a little about how GRPO will affect this token over time. If the downstream performance of the model never changes, then 

Running that oracle experiment is a little expensive, though. But here's a shortcut--just pick "To " every time. You'll get xx.x%

The oracle predicts correctness, and that in turn gives us an idea of what GRPO is going to do to this token if visited heavily. 

The most productive pathways are going to win








**Openers Matter a Lot**

The nucleus means that the model actually has a finite "prediction tree" that you could in theory walk--but even with these small nuclei it's enormous. 

The pathways must share plenty of machinery, such that applying RL to one branch node impacts its decisions on others as well.

There is one part of the sequence that we're going to visit repeatedly, though--the starting token in the model's response. 

We're only going to walk a tiny fraction of the possible rollouts for a given problem. Intuitively, though, the more times we visit a branch, and the 