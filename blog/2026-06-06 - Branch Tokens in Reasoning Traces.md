<!-- code -->
```python
# Defer annotation evaluation so helpers can be defined in their own cells
# above the Setup cell (which is where ``Path`` etc. are actually imported).
# In notebook execution this doesn't matter -- cells run top-to-bottom -- but
# running the file as a plain script otherwise hits a def-time NameError on
# the ``path: Path`` annotation of ``_write_html_file``.
from __future__ import annotations
```

<!-- md -->
Check for Colab vs. script.

This file can also be run as a script, so we need to guard some actions to only happen when we're running from within a Colab Notebook.

<!-- code -->
```python
try:
    from google.colab import userdata
    from IPython import get_ipython
    is_colab = get_ipython() is not None
except ImportError:
    is_colab = False
```

<!-- md -->
# ▂▂▂▂▂▂▂▂▂▂▂▂

<!-- md -->
# 

<!-- md -->
geometry/617
"The coordinates of a parallelogram are (5, 3), (6, 8), (7, 4) and $(x, y)$ and $x > 7$. What is the value of $x + y$?"

"answer": "17"

Number Theory 515
"What is the smallest positive perfect cube that can be written as the sum of three consecutive integers?"

Answer: 27

(Because $3^3 = 27$ and $8 + 9 + 10 = 27$)

- Language models predict the next token by computing logits over their entire vocabulary--i.e., for every word, there's a calculated value that reflects the model's confidence in that choice.
- The SoftMax function turns those logits into probabilities, and we pick the next token "at random", except weighted by those probabilities. 
- When you look at the actual probabilities, however, particularly in reasoning oriented tasks, it's surprisingly more deterministic.
- The token probabilities don't gently slope away towards less-and-less likely choices. Instead the model typically defines a sharp cutoff--a probability _cliff_. 
- Additionally, when we pick the next token, we don't actually consider the full vocabulary--we use a heuristic to try and detect that cliff called "nucleus sampling". 
- It places a hard cutoff on which tokens we actually consider, and it's defined by setting the "top-p" value. For reasoning tasks, a standard top-p value is 0.95, which means that we accumulate the most probable tokens until we've reached a total probability of 0.95. 
* There are a couple things that are quite striking about these token "nuclei".
* (1) - Most of them only have 1 token in them--we're "sampling" from a pool of 1.
* (2) - The non "singleton" nuclei often have surprisingly few choices in them.

The below illustration shows Qwen3-8B's nuclei for the start of its response to the question "what is 67 times 153?" For each non-singleton token, you'll see the list of choices below it, and their probabilities assigned by the model. Each of those represents a possible branch we could have taken while following this specific path.

> Pick a Math question, challenging but terse.
> Get the html visualizer working again.
> Get the probabilities correct and use nucleus, but just sample greedy

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