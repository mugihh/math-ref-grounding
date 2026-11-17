### Breakdown by candidate count (R@1)

| Method | 1-5 | 6-10 | 11-15 | 16+ |
|---|---|---|---|---|
| Random | 0.307 ± 0.020 | 0.136 ± 0.020 | 0.078 ± 0.012 | 0.035 ± 0.010 |
| Class-prior | 0.327 ± 0.008 | 0.145 ± 0.009 | 0.078 ± 0.007 | 0.073 ± 0.009 |
| CLIP dual-encoder | 0.414 | 0.228 | 0.158 | 0.093 |
| CLIP dual-encoder + LaTeX | 0.574 | 0.382 | 0.257 | 0.180 |
| GPT-5.4 | 0.768 | 0.483 | 0.362 | 0.267 |
| GPT-5.4 + LaTeX | 0.859 | 0.748 | 0.625 | 0.517 |
| Qwen2.5-VL-7B | 0.539 | 0.256 | 0.178 | 0.081 |
| Qwen2.5-VL-7B + LaTeX | 0.672 | 0.505 | 0.353 | 0.326 |

*R@1 per candidate count bucket. ± is the standard deviation over 5 seeds.*
