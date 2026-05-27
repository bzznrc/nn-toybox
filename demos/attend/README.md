# attend

Tiny self-attention on synthetic subject-verb agreement sentences.

Default display: `python -m nn_toybox.display --demo attend`

Headless run: `python -m nn_toybox.run --demo attend --steps 1000`

Example:

```text
the dogs near the cat [MASK] noisy
near the cats , the dog [MASK] noisy
today the dog behind the cats [MASK] loud
```

The target is `is` or `are` based on the subject, even when the closer noun is a misleading distractor.

This is not language modeling and not a full transformer. It is a small toy that shows one useful attention idea: the model learns which word matters for the current decision.
