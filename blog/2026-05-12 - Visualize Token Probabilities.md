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
# New Visualizations

<!-- md -->
Let's create a visualization of the token probabilities and "branch decisions" in Qwen's response to a math problem.

Influential paper: _"Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?"_ ([pdf](https://arxiv.org/pdf/2504.13837))

- RL closes bad branches and promotes good ones.
    - "More efficient sampling" = Good response in fewer tries.

<!-- md -->
<img src='https://lh3.googleusercontent.com/d/1QIod5FbQ1ixYmGXYWxoyBTe1I53gh_7u' alt='Diagram comparing base model and RLVR model branching trees for Problem A, illustrating how RL prunes branches to reach correct answers more efficiently.' width='1330' />

<!-- md -->
But it also tends to lose access to good responses in the process:

<!-- md -->
<img src='https://lh3.googleusercontent.com/d/1rKHyzGlsC1AjNZs2yxzYYIlY8VYPTVqA' alt='Diagram comparing base model and RLVR model branching trees for Problem B, illustrating how RL reduces the scope of the model's reasoning capacity by cutting off branches that lead to correct answers.' width='1350' />

<!-- md -->
They demonstrate this by sampling a very large number of responses (256).

The below plot shows accuracy on a math benchmark (y-axis) given the number of responses generated per question (x-axis). i.e., we treat the problem as correctly answered so long as one of the 'k' responses is correct.

The black line is the base model, and the colored lines are GRPO-tuned models. (They show performance at 3 checkpoints to illustrate how further tuning results in more lost responses)

The left side of the plot demonstrates the benefit of GRPO--the model produces correct answers in far fewer tries.

The far right end of the plot shows how, if we generate 256 answers, the base model outperforms the GRPO-tuned models, and the gap gets larger the more steps you tune for.

<!-- md -->
<img src='https://lh3.googleusercontent.com/d/187b8HTlvjHzpOLxhW1Bmh_KZ-YNaa2Ji' alt='Pass@k coverage curves on Omni-MATH-Train comparing Qwen2.5-7B base model to GRPO-tuned checkpoints at steps 150, 300, and 450; base model overtakes GRPO checkpoints at high k.' width='875' />

<!-- md -->
In this Notebook, we'll generate responses to a math problem, and visualize the model's output by showing the probability it assigns to the tokens in the sequence.

> "What is 145 times 37?"

Below is partial response, where:
1. The probability the model assigns to a token is written below it, and is reflected in the shading.
2. The top row shows our generated response; these tokens are also highlighted with a dark blue border.
3. For a token where there were multiple options, the alternates are listed below it.

<!-- md -->
<img src='https://lh3.googleusercontent.com/d/1K3-LGrYKb4uupvgbThA1olMYE8coIZWP' alt='Example token-by-token probability visualization showing one row of generated tokens with their assigned probabilities and alternate-token chips beneath each position.' width='2520' />

<!-- md -->
To generate a variety of responses, whenever we choose the next token, we select one at random, using the model's assigned probabilities.

Although every token in the vocabulary has a non-zero probability, our sampling methodology eliminates almost all of them.

In the visualizations at each token position, we list the possible token choices (displaying at most 10).

This reveals a couple fascinating insights:

1. At most token positions, the model is so certain of its prediction that we will never predict anything other than its top choice.
2. When the model is "uncertain", there are still typically only a few possible choices.

We can think of these "uncertain" tokens as branch points in the generation process, creating a tree of possible responses. The tree is massive, but finite.

RL is constrained to operate within this tree. It can only:
1. Adjust the relative probabilities of the branches
2. Cut off branches entirely.

It generally can not discover new branches.

> This last statement would be interesting to explore further--since decreasing the probability of a token must increase the probabilities of all others, can we contrive an example where this side effect causes a new token to cross the threshold and join the pool of options?

In this Notebook, we'll explore all of the above through examples.

<!-- md -->


<!-- md -->
Visualizers include
- formatting for displaying individual tokens with HTML, and using shading to convey prob.
- truncate after the first 10.
- Under each token, show the list of possible alternates (using the same nice formatting).
   - By alternate, I'm thinking to include the tokens that fall within p < 0.99.
       - I suspect that for most tokens the number of alternates is modest. But we should cap it at 10 (top token plus nine alternates).

Includes a heatmap colorbar for the token probabilities. - light blue (for low probability) to dark blue (for ~1.0).

- Column view makes it clear whether sampled token was top choice or not.

<!-- md -->
**Purpose**

- It should show how some token choices are "neutral"--all roughly the same prob, whereas others have more interesting distinction.
- If we just use the same color for all tokens, then the "100% confident" tokens will stand out for:
    - Being solid dark blue
    - Not having any alternate tokens in the column.

<!-- md -->
**Next Visualization**

- See if we can identify the problem token.
- If we fix it, what do we get?
   - Idea: Pick a branch, set it up to generate, e.g., 64 responses per branch choice and see how many are correct.

<!-- md -->
# ▂▂▂▂▂▂▂▂▂▂▂▂

<!-- md -->
# Helper Functions

<!-- md -->
**Styling helpers**

The color scale uses a `sqrt(p)` transform so that low probabilities (where most of the interesting alternates live) get more of the visible range. The hue stays at ~210 (blue); lightness sweeps 96 -> 32 and saturation 25 -> 80 as `sqrt(p)` goes 0 -> 1. Text flips to white once the background is dark enough to need contrast.

Each chip also shows the probability as small monospace text under the token, so the exact value is always legible even when two chips look similar.

<!-- code -->
```python
def _blue_at(intensity):
    """HSL blue at a normalized intensity in [0, 1]."""
    intensity = max(0.0, min(1.0, intensity))
    l = 96 - 64 * intensity   # lightness 96 -> 32
    s = 25 + 55 * intensity   # saturation 25 -> 80
    return f"hsl(210, {s:.0f}%, {l:.0f}%)"


RED_PROB_THRESHOLD = 1e-3  # below this we shade the chip red instead of blue


def prob_to_blue_style(p):
    p = max(0.0, min(1.0, float(p)))
    # Below the threshold we leave the blue scale entirely and shade the chip
    # a soft red. The point is to flag "this is essentially p~=0" picks --
    # which usually only happen on forced/teacher-forced chosen tokens that
    # the scoring model would never have sampled. A single shade is used
    # rather than a sub-gradient because the reader gains no information
    # from distinguishing p=1e-4 vs. p=1e-7.
    if p < RED_PROB_THRESHOLD:
        return "background-color: hsl(0, 75%, 88%); color: #7f1d1d;"
    intensity = p ** 0.5
    bg = _blue_at(intensity)
    text_color = "#ffffff" if intensity > 0.6 else "#1e293b"
    return f"background-color: {bg}; color: {text_color};"


def render_token_chip(token_str, prob, is_chosen):
    """Render a single chip. ``token_str`` is the already-decoded display
    string (the caller is responsible for ``tokenizer.decode`` + the
    "\\n" -> "↵" substitution). Making the helper tokenizer-agnostic lets the
    same renderer handle columns built from different vocabularies (e.g.
    base Qwen vs. Gemma) without swapping a global tokenizer."""
    tok = html.escape(token_str).replace(" ", "&nbsp;")
    if tok == "":
        tok = "&nbsp;"
    bg = prob_to_blue_style(prob)
    border = (
        "border: 2px solid #1e3a8a;"
        if is_chosen
        else "border: 1px solid rgba(15, 23, 42, 0.18);"
    )
    prob_text = f"{prob:.2f}" if prob >= 0.005 else f"{prob:.0e}"
    return (
        f"<span title=\"p={prob:.4f}\" style=\"{bg} {border} "
        f"display:inline-flex; flex-direction:column; align-items:center;"
        f" border-radius:6px; padding:3px 6px;"
        f" font-family:monospace; line-height:1.15; min-width:36px;\">"
        f"<span style=\"font-size:13px;\">{tok}</span>"
        f"<span style=\"font-size:8px; opacity:0.75; margin-top:1px;\">"
        f"{prob_text}</span>"
        f"</span>"
    )
```

<!-- md -->
**HTML export**

`visualize_branches` slices the captured `columns` and renders them as a wrapping flex grid. By default it renders all columns from `start` to the end, auto-wrapping into multiple rows once a row would exceed `max_row_width_px` (default 1024). The vertical `row-gap` keeps the wrapped rows visually separated.

A horizontal gradient strip labels the probability scale (0.0 on the left, 1.0 on the right). It's sampled from `_blue_at(sqrt(p))` so the bar's shading matches the chips. Below it, each generation step is a vertical column: the chosen token sits at the top so they line up in a row, and alternates drop down beneath in descending probability order. When the chosen token is the model's top-1 the column tapers from dark to light; when it isn't, the top chip is visibly lighter than the second one.

<!-- code -->
```python
DEFAULT_MAX_ROW_WIDTH_PX = 1024

_html_output_seq = 0


def _write_html_file(path: Path, body_fragment: str, title: str = "Token probabilities") -> Path:
    """Wrap fragment in a minimal UTF-8 document and write to disk."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_title = html.escape(title, quote=True)
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 16px; background: #f8fafc; }}
</style>
</head>
<body>
{body_fragment}
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")
    return path


def _build_colorbar_html():
    # Sample the same sqrt-mapped blue scale we use for chips, so the bar
    # accurately previews the color a given probability will produce.
    stops = []
    for i in range(11):
        p = i / 10.0
        stops.append(f"{_blue_at(p ** 0.5)} {i * 10}%")
    gradient = ", ".join(stops)

    red_swatch = (
        "<span style=\"display:inline-block; width:14px; height:12px;"
        " border-radius:3px; border:1px solid rgba(15, 23, 42, 0.18);"
        " background-color: hsl(0, 75%, 88%);\"></span>"
    )
    return (
        "<div style='display:flex; align-items:center; gap:8px;"
        " margin:8px 0 14px 0; font-size:12px; color:#334155;'>"
        "<span>Token probability</span>"
        "<span>0.0</span>"
        f"<span style=\"display:inline-block; width:240px; height:12px;"
        f" border-radius:3px; border:1px solid rgba(15, 23, 42, 0.18);"
        f" background: linear-gradient(to right, {gradient});\"></span>"
        "<span>1.0</span>"
        f"<span style='margin-left:16px;'>{red_swatch}</span>"
        f"<span>p &lt; {RED_PROB_THRESHOLD:g}</span>"
        "</div>"
    )


def visualize_branches(
    start=0,
    count=None,
    max_row_width_px=DEFAULT_MAX_ROW_WIDTH_PX,
    cols=None,
    output_path=None,
    title=None,
):
    """Build token-column HTML and write a standalone UTF-8 document to disk.

    Args:
        start: First column index to render.
        count: Number of columns to render. `None` means "to the end".
        max_row_width_px: Container width (px) before wrapping to a new row.
        cols: Optional override of the captured `columns` list.
        output_path: Destination `.html` path. If ``None``, writes next to this
            script as ``visualize_token_probs_NNN.html`` (incrementing per call).
        title: Optional ``<title>`` for the HTML document.
    """
    global _html_output_seq

    cols = columns if cols is None else cols
    end = len(cols) if count is None else start + count
    sliced = cols[start:end]
    if not sliced:
        fragment = "<p><em>No tokens in this slice.</em></p>"
        actual_end = start
    else:
        column_htmls = []
        for column in sliced:
            chips = "".join(
                render_token_chip(tok_str, p, is_chosen)
                for tok_str, p, is_chosen in column
            )
            # `flex:0 0 auto` so columns keep their natural width when the row wraps.
            column_htmls.append(
                "<div style='display:flex; flex-direction:column; gap:3px;"
                " align-items:stretch; flex:0 0 auto;'>"
                + chips
                + "</div>"
            )

        # gap: <row-gap> <column-gap>
        row_html = (
            f"<div style='display:flex; flex-flow:row wrap; gap:16px 6px;"
            f" align-items:flex-start; max-width:{max_row_width_px}px;'>"
            + "".join(column_htmls)
            + "</div>"
        )

        actual_end = start + len(sliced)
        header_html = (
            "<div style='margin:0 0 6px 0; font-size:12px; color:#475569;'>"
            f"Tokens [{start}, {actual_end})"
            "</div>"
        )
        fragment = header_html + _build_colorbar_html() + row_html

    if output_path is None:
        _html_output_seq += 1
        out = (
            Path(__file__).resolve().parent
            / "html"
            / f"visualize_token_probs_{_html_output_seq:03d}.html"
        )
    else:
        out = Path(output_path)

    doc_title = title if title is not None else f"Token probabilities [{start}, {actual_end})"
    _write_html_file(out, fragment, title=doc_title)
    print(f"Wrote HTML: {out}")

    # When running in Colab/Jupyter, also render the fragment inline so the
    # reader doesn't have to open the saved file. IPython is guaranteed to be
    # importable here because that's how `is_colab` got set to True above.
    if is_colab:
        from IPython.display import HTML, display
        display(HTML(fragment))

    return out
```

<!-- md -->
# ▂▂▂▂▂▂▂▂▂▂▂▂

<!-- md -->
# Visualizing Branches

<!-- md -->
**Setup**

<!-- code -->
```python
import html
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
#model_name = "google/gemma-2-2b-it"
#model_name = "google/gemma-3-4b-it"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
)
```

<!-- md -->
**Column-building helper**

Every visualization in this notebook is built from the same per-step shape:

- A forced/chosen token (its decoded string + its probability under whichever
  model we're scoring with).
- That step's "nucleus" of alternates -- tokens in descending probability
  order whose cumulative mass reaches `NUCLEUS_P`, dropping anything below
  `MIN_ALT_PROB`, capped at `MAX_PER_COL` chips total.

`build_columns` is tokenizer-agnostic, so the same helper produces columns
for Qwen-on-its-own-rollout, Qwen-on-Gemma's text, etc. It always returns
the parallel `columns_ids` list too -- block 1 needs it to look up new
probabilities under a second model later; other callers just ignore it.

<!-- code -->
```python
NUCLEUS_P    = 0.9   # Cumulative probability threshold for alternates
MIN_ALT_PROB = 0.01  # Hide alternates with raw probability below this
MAX_PER_COL  = 10    # Chosen + up to 9 alternates per column


def build_columns(probs, chosen_ids, tokenizer):
    """Build per-step visualization columns.

    Args:
        probs:      [T, V] indexable of per-step probability distributions.
                    Must satisfy ``T >= len(chosen_ids)``.
        chosen_ids: list[int] forced token ids; one column produced per id.
        tokenizer:  used only for ``tokenizer.decode([tid])``.

    Returns:
        ``(columns, columns_ids)``, parallel lists of length ``len(chosen_ids)``.
        ``columns[i]`` is ``[(tok_str, prob, is_chosen), ...]`` starting with
        the chosen token followed by nucleus alternates in descending order.
        ``columns_ids[i]`` is the parallel list of raw token ids.
    """
    columns     = []
    columns_ids = []
    for step, chosen_id in enumerate(chosen_ids):
        step_probs  = probs[step]
        chosen_prob = step_probs[chosen_id].item()
        chosen_str  = tokenizer.decode([chosen_id]).replace("\n", "↵")

        sorted_probs, sorted_ids = torch.sort(step_probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=0)
        # Include every token up to (and including) the one that crosses the threshold.
        nucleus_size = max(int((cum < NUCLEUS_P).sum().item()) + 1, 1)

        column     = [(chosen_str, chosen_prob, True)]
        column_ids = [chosen_id]
        for tid, p in zip(sorted_ids[:nucleus_size].tolist(),
                          sorted_probs[:nucleus_size].tolist()):
            if tid == chosen_id:
                continue
            if p < MIN_ALT_PROB:
                break  # probs are sorted descending -- everything after is smaller
            column.append((tokenizer.decode([tid]).replace("\n", "↵"), p, False))
            column_ids.append(tid)
            if len(column) >= MAX_PER_COL:
                break
        columns.append(column)
        columns_ids.append(column_ids)

    return columns, columns_ids
```

<!-- md -->
**Prompt**

> What is 145 times 37?

<!-- code -->
```python
prompt = "What is 145 times 37?"

messages = [{"role": "user", "content": prompt}]
chat_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
prompt_len = inputs["input_ids"].shape[1]
```

<!-- md -->
**Generate and capture branch data**

We use the same `gen_kwargs` as the reference notebook (no `top_k` override — Qwen's `generation_config` supplies the default of 20) so that with `seed=42` we reproduce the reference's first rollout exactly.

For the visualization we then run a separate teacher-forcing pass over the sampled sequence to recover the model's **raw** probability distribution at each generated step. That number is the model's actual confidence in a token, independent of `top_k`, `top_p`, or `repetition_penalty` — which is what we want a chip's shade to represent.

For each generated position we keep:

- The chosen token id and its raw probability.
- The "nucleus" of alternates: tokens sorted by descending raw probability whose cumulative mass reaches `NUCLEUS_P`, then dropping anything below `MIN_ALT_PROB`, capped at `MAX_PER_COL` entries total (chosen + up to 9 alternates).

We generate 64 tokens so we can render four cells of 16 tokens each.

<!-- code -->
```python
MAX_NEW_TOKENS = 512  # Total tokens generated

# Match the reference notebook's generation kwargs exactly so seed=42 reproduces
# its first rollout. top_k is intentionally not set -- it defaults to 20 from
# Qwen's generation_config.
gen_kwargs = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}
```

<!-- md -->
### Base Model

<!-- md -->
We'll use a consistent seed for our examples (42).

Here is what the Qwen 2.5 base model produces.

<!-- code -->
```python
torch.manual_seed(42)
with torch.no_grad():
    output_ids = model.generate(**inputs, **gen_kwargs)

# Teacher-forcing pass: feed the full sampled sequence back through the model
# in one shot to read the *raw* softmax at each generated step.
with torch.no_grad():
    teacher_logits = model(output_ids).logits  # [1, prompt_len + N, vocab]

# logits at position t predict the token at position t+1, so the distribution
# for the first generated token lives at index prompt_len - 1.
gen_logits = teacher_logits[0, (prompt_len - 1):]
raw_probs  = torch.softmax(gen_logits.float(), dim=-1)  # [N+1, vocab]

generated_ids = output_ids[0, prompt_len:].tolist()

# `columns_ids` is kept around so the FT re-scoring pass below can look up
# new probabilities by id without re-decoding. The other capture blocks
# don't need it and just drop it on the floor.
columns, columns_ids = build_columns(raw_probs, generated_ids, tokenizer)

print(f"Captured {len(columns)} columns; alternates per column:",
      [len(c) for c in columns])
print("Response:")
print(tokenizer.decode(generated_ids, skip_special_tokens=True))
```

<!-- md -->
**Render the full sequence**

<!-- code -->
```python
visualize_branches(
    output_path=Path(__file__).parent / "html" / "01_base_qwen.html",
    title="Qwen 2.5 0.5B base — its own rollout, raw probabilities",
)
```

<!-- md -->
**TODO** - Let's include a markdown-rendered output as well.

<!-- md -->
Again,
- The model's response is in the dark-blue-outlined tokens.
- We'll include up to 10 alternates if they exist; in this example there are never more 6 (at the token "Therefore").

<!-- md -->
<img src='https://lh3.googleusercontent.com/d/19I7JDxYLbnUmawmhAfJZrUEozRriMkhr' alt='Full token-probability visualization of the base Qwen 2.5 0.5B model's rollout for '145 times 37', showing tokens, probabilities, and alternates across multiple lines.' width='2677' />

<!-- md -->
## After Fine-Tuning

<!-- md -->

We're not generating new tokens here. Instead we ask: *if the GRPO-trained
model had been given exactly this prefix, how confident would it now be in
each of the base-model's chosen tokens and alternates?* The layout is
identical to the base rollout — same columns, same tokens — but each chip
is re-shaded with the fine-tuned model's probability for that token.

<!-- code -->
```python
# Stash everything we still need from the base rollout before tearing the
# base model down.
base_output_ids    = output_ids
base_prompt_len    = prompt_len
base_columns       = columns
base_columns_ids   = columns_ids
base_generated_ids = generated_ids

# Free the base model's GPU memory.
import gc

del model
gc.collect()
torch.cuda.empty_cache()

# Load the GRPO-trained checkpoint. Same tokenizer family as base, so we can
# reuse `tokenizer` and feed base_output_ids straight in.
ft_model = AutoModelForCausalLM.from_pretrained(
    "ChrisMcCormick/qwen2.5-0.5b-grpo-arithmetic",
    dtype=torch.bfloat16,
    device_map="auto",
)

# Teacher-force the base-model's full sequence through the FT model in one
# shot. Position t's logits predict the token at position t+1, so the
# distribution for the first generated token lives at index prompt_len - 1
# (matching the base capture above).
with torch.no_grad():
    ft_teacher_logits = ft_model(base_output_ids).logits  # [1, prompt_len + N, vocab]
ft_gen_logits = ft_teacher_logits[0, (base_prompt_len - 1):]
ft_probs      = torch.softmax(ft_gen_logits.float(), dim=-1)  # [N, vocab]

# Re-score: same column structure (same chosen token, same alternate ids,
# same is_chosen flags) but each prob is now looked up under ft_probs.
ft_columns = [
    [(tok_str, ft_probs[step, tid].item(), is_chosen)
     for (tok_str, _, is_chosen), tid in zip(col, col_ids)]
    for step, (col, col_ids) in enumerate(zip(base_columns, base_columns_ids))
]

print(f"Re-scored {len(ft_columns)} columns under the fine-tuned model.")
```

<!-- code -->
```python
visualize_branches(
    cols=ft_columns,
    output_path=Path(__file__).parent / "html" / "02_base_rollout_ft_scored.html",
    title="Base rollout — re-scored by the GRPO fine-tuned model",
)
```

<!-- md -->
## FT-Generated, Base-Scored

Flip the perspective: let the **fine-tuned** model do the sampling, then
teacher-force its response through the base model. Each chip's chosen
token is what GRPO picked; its shade is how confident the *base* model
would have been in that pick. Alternates are the base model's own top
choices at that step. Light chips next to darker alternates highlight
the steps where GRPO learned to prefer a token base would have been
hesitant to sample.

<!-- code -->
```python
# FT model is still loaded from the previous section -- sample its own
# rollout while we have it.
torch.manual_seed(42)
with torch.no_grad():
    ft_output_ids = ft_model.generate(**inputs, **gen_kwargs)
ft_prompt_len    = inputs.input_ids.shape[1]
ft_generated_ids = ft_output_ids[0, ft_prompt_len:].tolist()

print("=== FT-generated response ===")
print(tokenizer.decode(ft_generated_ids, skip_special_tokens=True))

# Free FT, reload Qwen base.
del ft_model
gc.collect()
torch.cuda.empty_cache()

qwen_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    dtype=torch.bfloat16,
    device_map="auto",
)

# Teacher-force the FT rollout through the base model to read base's raw
# probability at each FT-chosen position.
with torch.no_grad():
    base_on_ft_logits = qwen_model(ft_output_ids).logits
base_on_ft_probs = torch.softmax(
    base_on_ft_logits[0, (ft_prompt_len - 1):].float(), dim=-1
)  # [N, vocab]

# Build columns: chosen = FT's tokens, alternates = base's top picks.
base_on_ft_columns, _ = build_columns(
    base_on_ft_probs, ft_generated_ids, tokenizer
)

print(f"Built {len(base_on_ft_columns)} columns "
      f"(chosen = FT's tokens, alternates = base's top picks).")
```

<!-- code -->
```python
visualize_branches(
    cols=base_on_ft_columns,
    output_path=Path(__file__).parent / "html" / "03_ft_rollout_base_scored.html",
    title="GRPO fine-tuned rollout — re-scored by Qwen 2.5 base",
)
```

<!-- md -->
## Gemma 2 2B

A different angle on "what does this rollout look like under Qwen?".
Instead of letting Qwen sample, we **generate the response from Gemma 2 2B**,
then re-tokenize Gemma's text under Qwen's vocabulary and teacher-force it
through the Qwen base model. The chosen-token chip in each column is
whatever Gemma's text dictates; the alternates are Qwen's own top picks.

The visual payoff: every Gemma-forced step that Qwen would essentially
never have sampled appears as a light chip sitting next to much darker
Qwen-preferred alternates. That's the "valid alternative response that the
base Qwen model would essentially never produce" view.

<!-- code -->
```python
# The previous section left ``qwen_model`` (= base Qwen) loaded. Drop it
# so we have headroom for Gemma; we'll reload Qwen again afterwards.
del qwen_model
gc.collect()
torch.cuda.empty_cache()

# === Generate Gemma's response (we only need the string out) ==================
gemma_tok   = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
gemma_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    dtype=torch.bfloat16,
    device_map="auto",
)

gemma_chat = gemma_tok.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
)
gemma_inputs = gemma_tok(gemma_chat, return_tensors="pt").to(gemma_model.device)

torch.manual_seed(42)
with torch.no_grad():
    gemma_out = gemma_model.generate(**gemma_inputs, **gen_kwargs)
gemma_response = gemma_tok.decode(
    gemma_out[0, gemma_inputs.input_ids.shape[1]:],
    skip_special_tokens=True,
)
print("=== Gemma 2 2B response ===")
print(gemma_response)

del gemma_model, gemma_tok
gc.collect()
torch.cuda.empty_cache()

# === Reload Qwen base and re-tokenize Gemma's response under Qwen's vocab =====
qwen_tok   = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
qwen_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    dtype=torch.bfloat16,
    device_map="auto",
)

qwen_chat = qwen_tok.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
)
qwen_prompt_len = qwen_tok(qwen_chat, return_tensors="pt").input_ids.shape[1]
full = qwen_tok(qwen_chat + gemma_response, return_tensors="pt").to(qwen_model.device)

# === Teacher-force Gemma's continuation through Qwen ==========================
with torch.no_grad():
    qwen_logits = qwen_model(**full).logits
qwen_probs = torch.softmax(
    qwen_logits[0, (qwen_prompt_len - 1):].float(), dim=-1
)  # [N, vocab]
forced_ids = full.input_ids[0, qwen_prompt_len:].tolist()
print(f"Gemma response re-tokenized under Qwen into {len(forced_ids)} tokens.")

# === Build columns: chosen = Gemma's forced id, alternates = Qwen's top picks ==
qwen_on_gemma_columns, _ = build_columns(qwen_probs, forced_ids, qwen_tok)

print(f"Built {len(qwen_on_gemma_columns)} columns "
      f"(chosen = Gemma's tokens, alternates = Qwen's top picks).")
```

<!-- code -->
```python
visualize_branches(
    cols=qwen_on_gemma_columns,
    output_path=Path(__file__).parent / "html" / "04_gemma_rollout_qwen_scored.html",
    title="Gemma 2 2B rollout — re-scored by Qwen 2.5 base",
)
```

<!-- md -->
## Gemma 3 4B scored by Base

<!-- md -->
Below is a response taken from the Gemma 3 4B model which we'll feed through the base Qwen model to get the probabilities.

<!-- code -->
```python
gemma_3_4b_response = \
"""145 times 37 can be calculated as follows:

145 * 37 = 145 * (30 + 7)
= 145 * 30 + 145 * 7
= 4350 + (145 * 7)

Now, 145 * 7 can be calculated as:
145 * 7 = (100 + 40 + 5) * 7
= 100 * 7 + 40 * 7 + 5 * 7
= 700 + 280 + 35
= 700 + 315
= 1015

So, 145 * 37 = 4350 + 1015
= 5365

Alternatively, we can use the standard multiplication method:
   145
 x 37
-------
   1015  (145 * 7)
  4350  (145 * 30)
-------
  5365

Therefore, 145 times 37 is 5365.

Final Answer: The final answer is $\boxed{5365}$"""
```

<!-- code -->
```python
# qwen_model, qwen_tok, qwen_chat and qwen_prompt_len are all still in scope
# from the Gemma 2 2B section above -- same prompt, same Qwen base. So we
# just tack the hardcoded response onto the chat-templated prefix and
# teacher-force it through Qwen. `.strip()` drops the leading/trailing
# newlines that come from the """...""" literal.
full = qwen_tok(
    qwen_chat + gemma_3_4b_response.strip(),
    return_tensors="pt",
).to(qwen_model.device)

with torch.no_grad():
    qwen_logits = qwen_model(**full).logits
qwen_probs = torch.softmax(
    qwen_logits[0, (qwen_prompt_len - 1):].float(), dim=-1
)  # [N+1, vocab]
forced_ids = full.input_ids[0, qwen_prompt_len:].tolist()
print(f"Gemma 3 4B response re-tokenized under Qwen into {len(forced_ids)} tokens.")

# Build columns: chosen = Gemma 3 4B's forced id, alternates = Qwen's top picks.
qwen_on_gemma3_columns, _ = build_columns(qwen_probs, forced_ids, qwen_tok)

print(f"Built {len(qwen_on_gemma3_columns)} columns "
      f"(chosen = Gemma 3 4B's tokens, alternates = Qwen's top picks).")
```

<!-- code -->
```python
visualize_branches(
    cols=qwen_on_gemma3_columns,
    output_path=Path(__file__).parent / "html" / "04b_gemma3_4b_qwen_scored.html",
    title="Gemma 3 4B response — re-scored by Qwen 2.5 base",
)
```

<!-- md -->
# ▂▂▂▂▂▂▂▂▂▂▂▂

<!-- md -->
# Solutions Lost by GRPO

<!-- md -->
Let's see how many of the base model's correct solutions are no longer accessible to the GRPO-tuned model.
We'll generate 32 responses to our challenging multiplication problem using the base model.
Then we'll scan the FT-model probability at every token of each response and flag the response
as **"lost"** if any single token falls below ``LOST_THRESHOLD`` (``5e-4``) -- under
standard nucleus / top-k sampling that token is effectively unreachable, so the entire
response containing it can no longer be sampled.

Why not perplexity? PPL is a geometric mean across all tokens, so a single catastrophically
low probability (e.g. FT giving p ~= 1e-4 to opening with "The") gets washed out by hundreds
of high-probability tokens around it. The min-probability-under-threshold check is a much
sharper indicator of "we've lost this branch".

We also run the tuned model 32 times to see if it gets a higher percentage correct, and then
do the same scan in reverse: FT-generated rolls scored under the base model, with a more
permissive ``NOVEL_THRESHOLD`` (``5e-3``). That second view explores the recent research
finding that RL doesn't actually discover responses that weren't already in the base model's
distribution -- if true, very few FT rolls should contain tokens base would reject.

<!-- code -->
```python
import math
import re
import statistics

prompt          = "What is 145 times 37?"
ANSWER          = 145 * 37   # 5365
N_SAMPLES       = 32
LOST_THRESHOLD  = 5e-4       # FT prob < this on any token => base branch is unreachable under FT
NOVEL_THRESHOLD = 5e-3       # base prob < this on any token => FT response was outside base's distribution
```

<!-- md -->
`is_correct` is a helper function to identify the last number in the text as the answer, and returns 1.0 if correct, 0.0 if wrong.

<!-- code -->
```python
_ALL_NUMS_RE = re.compile(
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)

def is_correct(completion, answer):
    """
    Identifies the last number in the text as the answer, and
    returns 1.0 if correct, 0.0 if wrong.
    """
    matches = _ALL_NUMS_RE.findall(completion)

    if not matches:
        return 0.0

    last_number = int(float(matches[-1]))

    return 1.0 if last_number == answer else 0.0
```

<!-- md -->
**Per-token response probabilities.** Teacher-forces a single
`[prompt + response]` token sequence through the given model and returns
the model's probability for each response token. The eos token is included
in the response (so the model is scored on its decision to stop); any
padding past eos is excluded.

Returning the full per-token list (rather than a single aggregate) lets us
compute PPL when we want it *and* scan for catastrophically-low tokens --
the latter is the actual signal we care about for "branch reachability".

<!-- code -->
```python
def response_token_probs(sample_ids_1d, model, prompt_len, eos_id):
    """Returns (probs, n_response_tokens) for one sample.

    ``probs[i]`` is ``model``'s probability for the i-th response token.
    """
    rest = sample_ids_1d[prompt_len:]
    eos_positions = (rest == eos_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        end = prompt_len + int(eos_positions[0]) + 1  # include the eos token
    else:
        end = len(sample_ids_1d)

    seq = sample_ids_1d[:end].unsqueeze(0).to(model.device)
    with torch.no_grad():
        logits = model(seq).logits[0, :-1].float()
    probs = torch.softmax(logits, dim=-1)
    # logits[t] predicts seq[t+1]; the response targets are seq[prompt_len:end]
    # and they're scored by probs[prompt_len - 1 : end - 1].
    target_probs = probs[prompt_len - 1 : end - 1].gather(
        1, seq[0, prompt_len:end].unsqueeze(-1)
    ).squeeze(-1)
    return target_probs.tolist(), int(end - prompt_len)


def ppl_from_probs(probs):
    """Geometric-mean perplexity from a list of per-token probabilities.
    Floor the input at 1e-30 so log() doesn't blow up on a zero."""
    if not probs:
        return float("nan")
    return math.exp(-sum(math.log(max(p, 1e-30)) for p in probs) / len(probs))
```

<!-- md -->
**Generate 32 base responses and score them under base Qwen.**

`qwen_model` is still the base Qwen from the Gemma 3 4B section above, so
we sample from it directly. `num_return_sequences=N_SAMPLES` runs the rolls
in parallel; with `do_sample=True` each row gets an independent draw.

<!-- code -->
```python
chat_text  = qwen_tok.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False, add_generation_prompt=True,
)
inputs     = qwen_tok(chat_text, return_tensors="pt").to(qwen_model.device)
prompt_len = inputs.input_ids.shape[1]
eos_id     = qwen_tok.eos_token_id

torch.manual_seed(42)
with torch.no_grad():
    base_samples = qwen_model.generate(
        **inputs, **gen_kwargs, num_return_sequences=N_SAMPLES,
    )  # [N_SAMPLES, prompt_len + L_max]
print(f"Generated {N_SAMPLES} base samples; "
      f"padded length = {base_samples.shape[1] - prompt_len}.")

base_results = []  # list of dicts: correct, base_probs, base_ppl, n_tokens, text
for i in range(N_SAMPLES):
    base_probs, n_tokens = response_token_probs(
        base_samples[i], qwen_model, prompt_len, eos_id
    )
    text    = qwen_tok.decode(base_samples[i, prompt_len:], skip_special_tokens=True)
    correct = is_correct(text, ANSWER)
    base_results.append({
        "correct":    correct,
        "base_probs": base_probs,
        "base_ppl":   ppl_from_probs(base_probs),
        "n_tokens":   n_tokens,
        "text":       text,
    })

n_base_correct = int(sum(r["correct"] for r in base_results))
print(f"Base: {n_base_correct}/{N_SAMPLES} correct "
      f"({100 * n_base_correct / N_SAMPLES:.1f}%).")
```

<!-- md -->
**Re-score the same 32 responses under the GRPO fine-tuned model.**

Both base + FT are ~0.5B at bf16, so we just load FT alongside `qwen_model`
rather than tearing the base down. That way `qwen_model` stays live for the
downstream Prompt Perplexity / Prompt Visualizations sections.

For each response we record the FT model's *per-token* probabilities, then
extract two views: the geometric-mean PPL (for context) and the per-token
scan -- min FT prob, count of tokens below `LOST_THRESHOLD`, and the index
/ decoded string of the first such "killer" token. A response is flagged
as "lost" iff its min FT prob falls below the threshold.

<!-- code -->
```python
ft_model = AutoModelForCausalLM.from_pretrained(
    "ChrisMcCormick/qwen2.5-0.5b-grpo-arithmetic",
    dtype=torch.bfloat16,
    device_map="auto",
)

for r, sample in zip(base_results, base_samples):
    ft_probs, _ = response_token_probs(sample, ft_model, prompt_len, eos_id)
    r["ft_probs"] = ft_probs
    r["ft_ppl"]   = ppl_from_probs(ft_probs)
    r["min_ft_p"] = min(ft_probs)
    r["n_below"]  = sum(1 for p in ft_probs if p < LOST_THRESHOLD)
    # Index (within the response) of the first token FT considers unreachable.
    first_kill = next(
        (i for i, p in enumerate(ft_probs) if p < LOST_THRESHOLD), -1
    )
    r["first_kill_idx"] = first_kill
    if first_kill >= 0:
        killer_id = int(sample[prompt_len + first_kill])
        r["killer_tok"] = qwen_tok.decode([killer_id]).replace("\n", "\\n")
    else:
        r["killer_tok"] = ""
    r["lost_by_ft"] = r["min_ft_p"] < LOST_THRESHOLD
```

<!-- md -->
**Generate 32 FT responses (for the correct-rate comparison).**

<!-- code -->
```python
inputs = inputs.to(ft_model.device)

torch.manual_seed(42)
with torch.no_grad():
    ft_samples = ft_model.generate(
        **inputs, **gen_kwargs, num_return_sequences=N_SAMPLES,
    )

ft_results = []
for i in range(N_SAMPLES):
    ft_probs, n_tokens = response_token_probs(
        ft_samples[i], ft_model, prompt_len, eos_id
    )
    text    = qwen_tok.decode(ft_samples[i, prompt_len:], skip_special_tokens=True)
    correct = is_correct(text, ANSWER)
    ft_results.append({
        "correct":  correct,
        "ft_ppl":   ppl_from_probs(ft_probs),
        "n_tokens": n_tokens,
        "text":     text,
    })

n_ft_correct = int(sum(r["correct"] for r in ft_results))
print(f"FT:   {n_ft_correct}/{N_SAMPLES} correct "
      f"({100 * n_ft_correct / N_SAMPLES:.1f}%).")
```

<!-- md -->
**Per-response reachability scan.**

For each base-generated response we show whether the FT model assigns
probability below `LOST_THRESHOLD` to *any* of its tokens (`lost?` = Y).
`min_ft_p` is the smallest per-token probability anywhere in the response;
`n<th` is the count of tokens below threshold; `killer` is the index and
decoded string of the *first* such token (i.e. the earliest place the
branch becomes unreachable). `base_PPL` / `FT_PPL` are kept for context
but no longer drive the analysis.

<!-- code -->
```python
print(f"\n=== Base responses, reachability under FT ===")
print(f"  ANSWER         = {ANSWER}")
print(f"  LOST_THRESHOLD = p < {LOST_THRESHOLD:g}")
print(f"  Sorted by min FT token probability (most-blocked first).")
print()
print(f"{'idx':>3s}  {'OK':>2s}  {'lost':>4s}  {'min_ft_p':>10s}  "
      f"{'n<th':>4s}  {'killer (idx: tok)':<24s}  "
      f"{'base_PPL':>8s}  {'FT_PPL':>8s}  {'len':>4s}  preview")
print("-" * 140)

ordered = sorted(enumerate(base_results), key=lambda p: p[1]["min_ft_p"])
for idx, r in ordered:
    lost    = "Y" if r["lost_by_ft"] else "N"
    preview = r["text"].replace("\n", " ").strip()[:50]
    if r["first_kill_idx"] >= 0:
        killer = f"{r['first_kill_idx']}: {r['killer_tok']!r}"
    else:
        killer = "-"
    print(f"{idx:>3d}  {int(r['correct']):>2d}  {lost:>4s}  "
          f"{r['min_ft_p']:10.2e}  {r['n_below']:>4d}  {killer:<24s}  "
          f"{r['base_ppl']:8.3f}  {r['ft_ppl']:8.3f}  "
          f"{r['n_tokens']:4d}  {preview}")

# Reachability rollup: how many of base's correct vs. wrong rolls did FT lose?
correct_lost = [r for r in base_results if r["correct"] == 1.0 and r["lost_by_ft"]]
correct_keep = [r for r in base_results if r["correct"] == 1.0 and not r["lost_by_ft"]]
wrong_lost   = [r for r in base_results if r["correct"] == 0.0 and r["lost_by_ft"]]
wrong_keep   = [r for r in base_results if r["correct"] == 0.0 and not r["lost_by_ft"]]

print(f"\nReachability under FT (any token with p < {LOST_THRESHOLD:g})")
print(f"  correct base responses lost: {len(correct_lost):2d}/{n_base_correct} "
      f"({len(correct_keep)} still reachable)")
print(f"  wrong   base responses lost: {len(wrong_lost):2d}/{N_SAMPLES - n_base_correct} "
      f"({len(wrong_keep)} still reachable)")

# min_ft_p distribution by correctness -- a robust summary that doesn't average
# away the catastrophic-token signal the way PPL does.
def _median(xs):
    return statistics.median(xs) if xs else float("nan")

correct_min = [r["min_ft_p"] for r in base_results if r["correct"] == 1.0]
wrong_min   = [r["min_ft_p"] for r in base_results if r["correct"] == 0.0]
print(f"\nmin FT prob per response (smaller => more blocked)")
print(f"  correct ({len(correct_min):2d}):  "
      f"min {min(correct_min):.2e}   median {_median(correct_min):.2e}   max {max(correct_min):.2e}")
print(f"  wrong   ({len(wrong_min):2d}):  "
      f"min {min(wrong_min):.2e}   median {_median(wrong_min):.2e}   max {max(wrong_min):.2e}")

print(f"\n=== Correct rates on '{prompt}' (answer = {ANSWER}) ===")
print(f"  Base: {n_base_correct}/{N_SAMPLES} ({100 * n_base_correct / N_SAMPLES:.1f}%)")
print(f"  FT:   {n_ft_correct}/{N_SAMPLES} ({100 * n_ft_correct / N_SAMPLES:.1f}%)")
```

<!-- md -->
## Solutions Discovered by GRPO?

Reverse direction: take the 32 FT-generated rolls and ask whether the base
model would assign sub-threshold probability to any of their tokens. Recent
research argues that RL doesn't actually invent new responses -- it only
narrows the sampling distribution onto things the base model already had
(small but non-negligible) probability of producing. If that's right, then
almost every FT roll should still be reachable under base, even at a more
permissive ``NOVEL_THRESHOLD`` than we used for the LOST scan above.

`qwen_model` is still loaded from earlier, so we can score `ft_samples`
directly through it.

<!-- code -->
```python
for r, sample in zip(ft_results, ft_samples):
    base_probs, _ = response_token_probs(sample, qwen_model, prompt_len, eos_id)
    r["base_probs"] = base_probs
    r["base_ppl"]   = ppl_from_probs(base_probs)
    r["min_base_p"] = min(base_probs)
    r["n_below"]    = sum(1 for p in base_probs if p < NOVEL_THRESHOLD)
    first_kill = next(
        (i for i, p in enumerate(base_probs) if p < NOVEL_THRESHOLD), -1
    )
    r["first_kill_idx"] = first_kill
    if first_kill >= 0:
        killer_id = int(sample[prompt_len + first_kill])
        r["killer_tok"] = qwen_tok.decode([killer_id]).replace("\n", "\\n")
    else:
        r["killer_tok"] = ""
    r["novel_under_base"] = r["min_base_p"] < NOVEL_THRESHOLD

print(f"\n=== FT responses, reachability under base Qwen ===")
print(f"  NOVEL_THRESHOLD = p < {NOVEL_THRESHOLD:g}")
print(f"  Sorted by min base token probability (most-novel first).")
print()
print(f"{'idx':>3s}  {'OK':>2s}  {'novel':>5s}  {'min_base_p':>10s}  "
      f"{'n<th':>4s}  {'killer (idx: tok)':<24s}  "
      f"{'base_PPL':>8s}  {'FT_PPL':>8s}  {'len':>4s}  preview")
print("-" * 140)

ordered = sorted(enumerate(ft_results), key=lambda p: p[1]["min_base_p"])
for idx, r in ordered:
    novel   = "Y" if r["novel_under_base"] else "N"
    preview = r["text"].replace("\n", " ").strip()[:50]
    if r["first_kill_idx"] >= 0:
        killer = f"{r['first_kill_idx']}: {r['killer_tok']!r}"
    else:
        killer = "-"
    print(f"{idx:>3d}  {int(r['correct']):>2d}  {novel:>5s}  "
          f"{r['min_base_p']:10.2e}  {r['n_below']:>4d}  {killer:<24s}  "
          f"{r['base_ppl']:8.3f}  {r['ft_ppl']:8.3f}  "
          f"{r['n_tokens']:4d}  {preview}")

correct_novel = [r for r in ft_results if r["correct"] == 1.0 and r["novel_under_base"]]
correct_known = [r for r in ft_results if r["correct"] == 1.0 and not r["novel_under_base"]]
wrong_novel   = [r for r in ft_results if r["correct"] == 0.0 and r["novel_under_base"]]
wrong_known   = [r for r in ft_results if r["correct"] == 0.0 and not r["novel_under_base"]]

print(f"\nFT responses outside base's distribution "
      f"(any token with p < {NOVEL_THRESHOLD:g} under base)")
print(f"  correct FT responses novel: {len(correct_novel):2d}/{n_ft_correct} "
      f"({len(correct_known)} were already reachable under base)")
print(f"  wrong   FT responses novel: {len(wrong_novel):2d}/{N_SAMPLES - n_ft_correct} "
      f"({len(wrong_known)} were already reachable under base)")

correct_min = [r["min_base_p"] for r in ft_results if r["correct"] == 1.0]
wrong_min   = [r["min_base_p"] for r in ft_results if r["correct"] == 0.0]
print(f"\nmin base prob per FT response (smaller => more outside base distribution)")
print(f"  correct ({len(correct_min):2d}):  "
      f"min {min(correct_min):.2e}   median {_median(correct_min):.2e}   max {max(correct_min):.2e}")
print(f"  wrong   ({len(wrong_min):2d}):  "
      f"min {min(wrong_min):.2e}   median {_median(wrong_min):.2e}   max {max(wrong_min):.2e}")
```

<!-- md -->
# ▂▂▂▂▂▂▂▂▂▂▂▂

<!-- md -->
# Prompt Perplexity

<!-- md -->
Out-of-distribution prompts may impact the model's performance.
The problem templates we used to generate our dataset did not take this into account.
Below are the templates and generation code we used.

<!-- code -->
```python
id_templates = [
    # Addition
    ("What is {} plus {}?", "+"),
    ("Calculate the sum of {} and {}.", "+"),
    ("What's the result of {} plus {}?", "+"),
    ("How much is {} plus {}?", "+"),

    # Subtraction
    ("What is {} minus {}?", "-"),
    ("Compute {} minus {}.", "-"),
    ("What's the result of {} minus {}?", "-"),
    ("How much is {} minus {}?", "-"),

    # Multiplication
    ("What is {} multiplied by {}?", "*"),
    ("Calculate the product of {} and {}.", "*"),
    ("What is the product of {} and {}?", "*"),
    ("How much is {} times {}?", "*"),

    # Division (always integer: dividend = quotient * divisor)
    ("What is {} divided by {}?", "/"),
    ("Calculate {} divided by {}.", "/"),
    ("What's the result of {} divided by {}?", "/"),
    ("How much is {} divided by {}?", "/"),
]
```

<!-- md -->
**Out-of-Domain Templates**

<!-- md -->
16 templates using **different phrasings**. Only used for **test_ood**.
Subtraction templates with reversed operand order ("subtract A from B"
means B − A) are marked with `"rev"`.

<!-- code -->
```python
ood_templates = [
    # Addition
    ("If you add {} and {}, what's the answer?", "+"),
    ("{} + {} = ?", "+"),
    ("Find the result of {} + {}.", "+"),
    ("What do you get when you add {} and {}?", "+"),

    # Subtraction (first two: normal; last two: reversed)
    ("{} - {} = ?", "-"),
    ("Find the result of {} - {}.", "-"),
    ("If you subtract {} from {}, what's left?", "-", "rev"),
    ("What remains when you take {} from {}?", "-", "rev"),

    # Multiplication
    ("If you multiply {} by {}, what do you get?", "*"),
    ("{} x {} = ?", "*"),
    ("Find the result of {} * {}.", "*"),
    ("What do you get when you multiply {} and {}?", "*"),

    # Division
    ("If you divide {} by {}, what do you get?", "/"),
    ("{} ÷ {} = ?", "/"),
    ("Find the result of {} / {}.", "/"),
    ("What do you get when you divide {} by {}?", "/"),
]

OP_LABELS = {
    "+": "Addition",
    "-": "Subtraction",
    "*": "Multiplication",
    "/": "Division",
}
```

<!-- md -->
### `get_random_problem`

<!-- md -->
`get_random_problem` handles all four operations. Division always produces
integer results by constructing `dividend = quotient * divisor`.

If `op` is given, only templates for that operator are used. Otherwise a
random template (any operator) is chosen.

<!-- code -->
```python
def get_random_problem(rng, templates, op=None,
                       max_operand_add=10000, max_operand_sub=10000,
                       max_operand_mult=50,
                       max_quotient=100, max_divisor=100):
    if op is not None:
        filtered = [t for t in templates if t[1] == op]
    else:
        filtered = templates

    entry = filtered[int(rng.integers(0, len(filtered)))]
    q, op_sym = entry[0], entry[1]
    reversed_operands = len(entry) > 2 and entry[2] == "rev"

    if op_sym == "+":
        val1, val2 = int(rng.integers(0, max_operand_add)), int(rng.integers(0, max_operand_add))
        answer = val1 + val2
        question = q.format(val1, val2)

    elif op_sym == "-":
        val1, val2 = int(rng.integers(0, max_operand_sub)), int(rng.integers(0, max_operand_sub))
        answer = (val2 - val1) if reversed_operands else (val1 - val2)
        question = q.format(val1, val2)

    elif op_sym == "*":
        val1, val2 = int(rng.integers(0, max_operand_mult)), int(rng.integers(0, max_operand_mult))
        answer = val1 * val2
        question = q.format(val1, val2)

    elif op_sym == "/":
        divisor = int(rng.integers(1, max_divisor))
        quotient = int(rng.integers(0, max_quotient))
        dividend = quotient * divisor
        answer = quotient
        if reversed_operands:
            question = q.format(divisor, dividend)
        else:
            question = q.format(dividend, divisor)

    else:
        raise ValueError(f"Unknown op: {op_sym}")

    return question, float(answer), op_sym
```

<!-- md -->
Quick sanity check.

<!-- code -->
```python
import math

import numpy as np

_rng_check = np.random.default_rng()

print("=== In-Domain examples ===")
for _op in ["+", "-", "*", "/"]:
    _q, _a, _ = get_random_problem(_rng_check, id_templates, op=_op)
    print(f"  {_op}: {_q}  ->  {int(_a)}")

print("\n=== Out-of-Domain examples ===")
for _op in ["+", "-", "*", "/"]:
    _q, _a, _ = get_random_problem(_rng_check, ood_templates, op=_op)
    print(f"  {_op}: {_q}  ->  {int(_a)}")
```

<!-- md -->
**Score each template under the base model**

Iterate through every (ID + OOD) template, fill in random operands using
the limits baked into `get_random_problem`, apply the chat template, and
compute the **content-token perplexity** under Qwen 2.5 base. The
"content" range is the user's question text only — not the surrounding
role-marker wrapper — so wrapper length doesn't dilute the signal. We
also report perplexity over the full chat-templated sequence for context.

A high content-PPL flags a phrasing the base model finds unusual; those
are candidates for "OOD prompts that may hurt model performance".

<!-- code -->
```python
# Sample one (question, op) per template using a fixed seed so the table is
# reproducible. Operand limits come from get_random_problem's defaults
# (additions 0-9999, multiplications 0-49, division quotient/divisor 0-99).
_rng = np.random.default_rng(0)

prompt_examples = []   # list[tuple[split, op, template, question]]
for _tpl in id_templates:
    _q, _, _op = get_random_problem(_rng, [_tpl])
    prompt_examples.append(("ID", _op, _tpl[0], _q))
for _tpl in ood_templates:
    _q, _, _op = get_random_problem(_rng, [_tpl])
    prompt_examples.append(("OOD", _op, _tpl[0], _q))

print(f"Scoring {len(prompt_examples)} prompts under: "
      f"{qwen_model.config.name_or_path}\n")

prompt_perplexities = []
for split, op, template, question in prompt_examples:
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Split the chat-templated text around the question so we can isolate
    # the content tokens. The wrapper bytes are identical across prompts so
    # they'd just dilute the per-template signal in an overall PPL.
    q_start = chat_text.index(question)
    prefix  = chat_text[:q_start]
    suffix  = chat_text[q_start + len(question):]

    pre_ids  = tokenizer(prefix,    return_tensors="pt", add_special_tokens=False).input_ids[0]
    con_ids  = tokenizer(question,  return_tensors="pt", add_special_tokens=False).input_ids[0]
    suf_ids  = tokenizer(suffix,    return_tensors="pt", add_special_tokens=False).input_ids[0]
    full_ids = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    assert len(pre_ids) + len(con_ids) + len(suf_ids) == len(full_ids), (
        f"BPE boundary effect on {question!r}: "
        f"{len(pre_ids)} + {len(con_ids)} + {len(suf_ids)} != {len(full_ids)}"
    )

    ids = full_ids.unsqueeze(0).to(qwen_model.device)
    with torch.no_grad():
        logits = qwen_model(ids).logits  # [1, L, V]
    log_probs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    nll = -log_probs.gather(1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)  # [L-1]

    # logits[t] predicts ids[t+1], so nll[t] is the loss on token t+1.
    # The content tokens (ids at positions [pre_len, pre_len + con_len)) are
    # scored at nll indices [pre_len - 1, pre_len - 1 + con_len).
    pre_len, con_len = len(pre_ids), len(con_ids)
    content_nll = nll[pre_len - 1 : pre_len - 1 + con_len]
    ppl_full    = math.exp(nll.mean().item())
    ppl_content = math.exp(content_nll.mean().item())

    prompt_perplexities.append(
        (split, op, template, question, ppl_content, ppl_full, len(full_ids))
    )

# Highest content-PPL first.
prompt_perplexities.sort(key=lambda r: r[4], reverse=True)

print(f"{'Split':5s}  {'Op':3s}  {'PPL_content':>11s}  {'PPL_full':>8s}  "
      f"{'L':>3s}  Question")
print("-" * 110)
for split, op, _, question, ppl_c, ppl_f, L in prompt_perplexities:
    print(f"{split:5s}  {op:3s}  {ppl_c:11.2f}  {ppl_f:8.2f}  {L:3d}  {question}")
```

<!-- md -->
## Prompt Visualizations

Same column layout as the response visualizations, but here we shade the
**prompt** tokens instead of the response. Each chip's chosen token is what
the prompt actually contains; its shade is how confident base Qwen would
have been to write that token at that position. Alternates are Qwen's own
top picks at the step. Light "chosen" chips next to darker alternates mark
places where the prompt's phrasing is genuinely out-of-distribution for the
base model -- often where the high-PPL templates above earned their score.

We render four hand-picked prompts spanning the high end of the PPL table.

<!-- code -->
```python
def build_prompt_columns(question, tokenizer, model):
    """Teacher-force a chat-templated `question` through `model` and return
    `build_columns` output covering only the user-question tokens (i.e. not
    the role-marker wrapper). Chosen = the prompt's actual token; alternates
    = the model's top picks at that position."""
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Split the chat-templated text around the question so we can isolate
    # just the content tokens. Mirrors the perplexity loop above.
    q_start = chat_text.index(question)
    prefix  = chat_text[:q_start]
    suffix  = chat_text[q_start + len(question):]

    pre_ids  = tokenizer(prefix,    return_tensors="pt", add_special_tokens=False).input_ids[0]
    con_ids  = tokenizer(question,  return_tensors="pt", add_special_tokens=False).input_ids[0]
    suf_ids  = tokenizer(suffix,    return_tensors="pt", add_special_tokens=False).input_ids[0]
    full_ids = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    assert len(pre_ids) + len(con_ids) + len(suf_ids) == len(full_ids), (
        f"BPE boundary effect on {question!r}: "
        f"{len(pre_ids)} + {len(con_ids)} + {len(suf_ids)} != {len(full_ids)}"
    )

    ids = full_ids.unsqueeze(0).to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0]  # [L, V]
    probs = torch.softmax(logits.float(), dim=-1)

    # logits[t] predicts ids[t+1], so the distribution for content token i
    # (at full_ids[pre_len + i]) lives at probs[pre_len - 1 + i].
    pre_len, con_len = len(pre_ids), len(con_ids)
    content_probs = probs[pre_len - 1 : pre_len - 1 + con_len]
    content_ids   = full_ids[pre_len : pre_len + con_len].tolist()

    return build_columns(content_probs, content_ids, tokenizer)


prompt_viz_examples = [
    "6175 ÷ 95 = ?",
    "How much is 33 times 0?",
    "Compute 6494 minus 9127.",
    "What remains when you take 5256 from 6471?",
]

for i, question in enumerate(prompt_viz_examples, start=1):
    cols, _ = build_prompt_columns(question, tokenizer, qwen_model)
    visualize_branches(
        cols=cols,
        output_path=Path(__file__).parent / "html" / f"05_prompt_{i:02d}.html",
        title=f"Prompt token probabilities — {question}",
    )
```

<!-- md -->
# ▂▂▂▂▂▂▂▂▂▂▂▂

<!-- md -->
# GSM8K

<!-- code -->
```python
prompt = \
"""
A carnival snack booth made \\(\\$50\\) selling popcorn each day. It made three times as much selling cotton candy. For a \\(5\\)-day activity, the booth has to pay \\(\\$30\\) rent and \\(\\$75\\) for the cost of the ingredients. How much did the booth earn for 5 days after paying the rent and the cost of ingredients?
"""
```

<!-- md -->
# ▂▂▂▂▂▂▂▂▂▂▂▂

<!-- md -->
## Gemma 3 4B

<!-- md -->
GSM8K:

```
A carnival snack booth made \(\$50\) selling popcorn each day. It made three times as much selling cotton candy. For a \(5\)-day activity, the booth has to pay \(\$30\) rent and \(\$75\) for the cost of the ingredients. How much did the booth earn for 5 days after paying the rent and the cost of ingredients?
```

<!-- md -->
```
Let $P$ be the amount made selling popcorn each day, and $C$ be the amount made selling cotton candy each day.
Given that the booth made \$50 selling popcorn each day, so $P = 50$.
The booth made three times as much selling cotton candy, so $C = 3 \times P = 3 \times 50 = 150$.
The total amount made each day is $P + C = 50 + 150 = 200$.
For a 5-day activity, the booth earns $5 \times (P+C) = 5 \times 200 = 1000$.
The booth has to pay \$30 rent and \$75 for the cost of ingredients.
The total cost is $30 + 75 = 105$.
The profit is the total earnings minus the total cost, so the profit is $1000 - 105 = 895$.

The booth made $50 per day selling popcorn.
The booth made $3 \times 50 = 150$ per day selling cotton candy.
The total earned per day is $50 + 150 = 200$.
For 5 days, the total earned is $5 \times 200 = 1000$.
The rent is $30$.
The cost of ingredients is $75$.
The total cost is $30 + 75 = 105$.
The profit is $1000 - 105 = 895$.

Final Answer: The final answer is $\boxed{895}$
```

<!-- code -->
```python
# visualize_branches()
```

