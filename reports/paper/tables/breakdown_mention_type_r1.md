### Breakdown by mention type (R@1)

| Method | implicit | explicit | unknown |
|---|---|---|---|
| Random | 0.165 ± 0.007 | 0.148 ± 0.013 | — |
| Class-prior | 0.203 ± 0.008 | 0.149 ± 0.004 | — |
| CLIP dual-encoder | 0.191 | 0.264 | — |
| CLIP dual-encoder + LaTeX | 0.138 | 0.476 | — |
| GPT-5.4 | 0.361 | 0.559 | — |
| GPT-5.4 + LaTeX | 0.568 | 0.783 | — |
| Qwen2.5-VL-7B | 0.254 | 0.304 | — |
| Qwen2.5-VL-7B + LaTeX | 0.396 | 0.525 | — |

*R@1 per mention type bucket. ± is the standard deviation over 5 seeds.*
