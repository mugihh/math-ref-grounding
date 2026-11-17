### Single-gold vs multi-gold

| Method | Single R@1 | Single R@3 | Single MRR | Multi R@1 | Multi R@3 | Multi MRR |
|---|---|---|---|---|---|---|
| Random | 0.152 ± 0.011 | 0.425 ± 0.016 | 0.358 ± 0.010 | 0.273 ± 0.072 | 0.600 ± 0.109 | 0.485 ± 0.045 |
| Class-prior | 0.165 ± 0.004 | 0.443 ± 0.006 | 0.371 ± 0.004 | 0.173 ± 0.020 | 0.473 ± 0.089 | 0.378 ± 0.018 |
| CLIP dual-encoder | 0.242 | 0.510 | 0.436 | 0.227 | 0.636 | 0.460 |
| CLIP dual-encoder + LaTeX | 0.375 | 0.603 | 0.534 | 0.364 | 0.682 | 0.562 |
| GPT-5.4 | 0.502 | 0.742 | 0.647 | 0.364 | 0.773 | 0.577 |
| GPT-5.4 + LaTeX | 0.720 | 0.901 | 0.821 | 0.591 | 0.864 | 0.741 |
| Qwen2.5-VL-7B | 0.289 | 0.531 | 0.465 | 0.318 | 0.545 | 0.468 |
| Qwen2.5-VL-7B + LaTeX | 0.488 | 0.715 | 0.631 | 0.364 | 0.773 | 0.587 |

*Recall@k accepts any gold id; MRR uses the highest-ranked gold. ± is the standard deviation over 5 seeds.*
