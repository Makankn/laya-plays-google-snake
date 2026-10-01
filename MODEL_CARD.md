# Snake-adapted Laya checkpoint

This is a full, loadable checkpoint for the `snake_laya` controller, [published on Hugging Face](https://huggingface.co/BoogieKn/laya-plays-google-snake) separately from the Git repository.

## Origin and adaptation

- **Base:** [`convaiinnovations/laya`, multilingual subfolder](https://huggingface.co/convaiinnovations/laya), published by Convai Innovations under Apache-2.0. The checkpoint retains its mmBERT-base encoder and tokenizer.
- **Adaptation:** The encoder is frozen while the typed-decision layers are trained on balanced `UP`/`RIGHT`/`DOWN`/`LEFT` choices from simulated Snake board states. The training and simulator code are in this repository. The shipped checkpoint's config records `snake-balanced-imitation-frozen-encoder`; it does not embed the exact hyperparameters of the original run.
- **Input/output:** Structured board facts and short legal-move descriptions go in; a typed-choice probability distribution over the directions comes out. This model does **not** interpret screenshots. Deterministic code handles perception and excludes unsafe actions.

The Hugging Face repository contains the full model weights, config, encoder config, tokenizer, model card, `LICENSE`, and `NOTICE`. Download it into `models/snake-laya-balanced/`:

```bash
uv run --extra laya hf download BoogieKn/laya-plays-google-snake --local-dir models/snake-laya-balanced
```

## Reproducible evaluation

On 2026-09-26, with the repository's current evaluator and local checkpoint:

```bash
uv run --extra laya snake-state eval-laya --model models/snake-laya-balanced --device cuda --episodes 10 --max-steps 160 --seed 10000 --validation-samples 0
```

The 10 fixed-seed simulation episodes averaged **12.6 apples per 160 steps** (best 15, worst 6). One episode spent 71 steps repeating positions, so looping is still a real failure mode. This is a simulator result, **not** a measured live Google Snake score or a claim that the agent cannot crash. The evaluation did not include vision errors, browser focus loss, occlusion, or real keypress timing.

To train a new local checkpoint with the included balanced curriculum:

```bash
uv run --extra laya snake-state train-laya --device cuda --output models/snake-laya-balanced
```

The CLI defaults to 1,200 synthetic curriculum samples, eight imitation epochs, seed 0, and a `3e-4` learning rate. A new run is not guaranteed to recreate these exact released weights.

## Intended use and limits

This checkpoint is meant for research and demonstration of fast typed decisions in a screenshot-driven controller. It was adapted to a 17×15 Google Snake board and compact English move descriptions. Its displayed probabilities are model outputs for the offered choices, not a calibrated probability of surviving a live game. A safety intervention, stale visual state, or late browser input can make the executed action differ from its top prediction. Do not use it for safety-critical control.

See [NOTICE](NOTICE) for upstream attribution and [LICENSE](LICENSE) for Apache-2.0 terms. The project is independent of Google and Convai Innovations.
