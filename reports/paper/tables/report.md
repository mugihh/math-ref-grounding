# Results

### Main results

| Method | R@1 | R@3 | MRR |
|---|---|---|---|
| Random | 0.153 ± 0.011 | 0.427 ± 0.016 | 0.359 ± 0.010 |
| Class-prior | 0.165 ± 0.004 | 0.443 ± 0.006 | 0.371 ± 0.004 |
| CLIP dual-encoder | 0.242 | 0.512 | 0.436 |
| CLIP dual-encoder + LaTeX | 0.375 | 0.604 | 0.534 |
| GPT-5.4 | 0.500 | 0.743 | 0.646 |
| GPT-5.4 + LaTeX | 0.719 | 0.901 | 0.820 |
| Qwen2.5-VL-7B | 0.289 | 0.531 | 0.465 |
| Qwen2.5-VL-7B + LaTeX | 0.486 | 0.716 | 0.630 |

*Mention-to-box ranking. ± is the standard deviation over 5 seeds.*

### Auxiliary spatial evaluation

| Method | Coverage | IoU | Med. dist | dist@0.05 | dist@0.10 |
|---|---|---|---|---|---|
| Random | 1.000 ± 0.000 | 0.186 ± 0.011 | 0.105 ± 0.003 | 0.302 ± 0.012 | 0.485 ± 0.010 |
| Class-prior | 1.000 ± 0.000 | 0.202 ± 0.003 | 0.096 ± 0.001 | 0.321 ± 0.002 | 0.511 ± 0.003 |
| CLIP dual-encoder | 1.000 | 0.279 | 0.083 | 0.388 | 0.544 |
| CLIP dual-encoder + LaTeX | 1.000 | 0.399 | 0.058 | 0.473 | 0.617 |
| GPT-5.4 | 1.000 | 0.539 | 0.000 | 0.630 | 0.751 |
| GPT-5.4 + LaTeX | 1.000 | 0.748 | 0.000 | 0.795 | 0.852 |
| Qwen2.5-VL-7B | 1.000 | 0.325 | 0.065 | 0.444 | 0.628 |
| Qwen2.5-VL-7B + LaTeX | 1.000 | 0.529 | 0.013 | 0.593 | 0.711 |

*Auxiliary spatial agreement between the selected gold candidate and the gold box; this is not detection evaluation. ± is the standard deviation over 5 seeds.*

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

### Breakdown by object type (R@1)

| Method | equation | span | term | figure | figure_part |
|---|---|---|---|---|---|
| Random | 0.183 ± 0.014 | 0.166 ± 0.011 | 0.127 ± 0.010 | 0.200 ± 0.088 | 0.133 ± 0.042 |
| Class-prior | 0.234 ± 0.002 | 0.192 ± 0.008 | 0.123 ± 0.005 | 0.000 ± 0.000 | 0.050 ± 0.008 |
| CLIP dual-encoder | 0.226 | 0.220 | 0.279 | 0.615 | 0.167 |
| CLIP dual-encoder + LaTeX | 0.212 | 0.323 | 0.535 | 0.154 | 0.083 |
| GPT-5.4 | 0.540 | 0.508 | 0.494 | 0.462 | 0.333 |
| GPT-5.4 + LaTeX | 0.850 | 0.736 | 0.697 | 0.462 | 0.347 |
| Qwen2.5-VL-7B | 0.358 | 0.340 | 0.218 | 0.385 | 0.097 |
| Qwen2.5-VL-7B + LaTeX | 0.708 | 0.471 | 0.489 | 0.154 | 0.028 |

*R@1 per object type bucket. ± is the standard deviation over 5 seeds.*

### Breakdown by object type (MRR)

| Method | equation | span | term | figure | figure_part |
|---|---|---|---|---|---|
| Random | 0.389 ± 0.012 | 0.375 ± 0.011 | 0.330 ± 0.009 | 0.411 ± 0.086 | 0.333 ± 0.027 |
| Class-prior | 0.415 ± 0.003 | 0.404 ± 0.006 | 0.333 ± 0.005 | 0.170 ± 0.001 | 0.217 ± 0.008 |
| CLIP dual-encoder | 0.420 | 0.429 | 0.455 | 0.737 | 0.349 |
| CLIP dual-encoder + LaTeX | 0.379 | 0.499 | 0.671 | 0.286 | 0.236 |
| GPT-5.4 | 0.680 | 0.655 | 0.639 | 0.614 | 0.512 |
| GPT-5.4 + LaTeX | 0.915 | 0.832 | 0.807 | 0.648 | 0.517 |
| Qwen2.5-VL-7B | 0.517 | 0.507 | 0.408 | 0.483 | 0.329 |
| Qwen2.5-VL-7B + LaTeX | 0.790 | 0.629 | 0.627 | 0.287 | 0.243 |

*MRR per object type bucket. ± is the standard deviation over 5 seeds.*

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

### Breakdown by mention type (MRR)

| Method | implicit | explicit | unknown |
|---|---|---|---|
| Random | 0.371 ± 0.006 | 0.354 ± 0.012 | — |
| Class-prior | 0.397 ± 0.008 | 0.360 ± 0.002 | — |
| CLIP dual-encoder | 0.384 | 0.458 | — |
| CLIP dual-encoder + LaTeX | 0.321 | 0.625 | — |
| GPT-5.4 | 0.535 | 0.694 | — |
| GPT-5.4 + LaTeX | 0.705 | 0.869 | — |
| Qwen2.5-VL-7B | 0.440 | 0.476 | — |
| Qwen2.5-VL-7B + LaTeX | 0.554 | 0.663 | — |

*MRR per mention type bucket. ± is the standard deviation over 5 seeds.*

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

### Breakdown by candidate count (MRR)

| Method | 1-5 | 6-10 | 11-15 | 16+ |
|---|---|---|---|---|
| Random | 0.561 ± 0.014 | 0.353 ± 0.013 | 0.254 ± 0.012 | 0.163 ± 0.013 |
| Class-prior | 0.576 ± 0.006 | 0.359 ± 0.005 | 0.260 ± 0.005 | 0.203 ± 0.009 |
| CLIP dual-encoder | 0.635 | 0.431 | 0.334 | 0.231 |
| CLIP dual-encoder + LaTeX | 0.732 | 0.543 | 0.420 | 0.319 |
| GPT-5.4 | 0.864 | 0.647 | 0.531 | 0.412 |
| GPT-5.4 + LaTeX | 0.921 | 0.841 | 0.750 | 0.676 |
| Qwen2.5-VL-7B | 0.711 | 0.455 | 0.351 | 0.193 |
| Qwen2.5-VL-7B + LaTeX | 0.801 | 0.649 | 0.518 | 0.444 |

*MRR per candidate count bucket. ± is the standard deviation over 5 seeds.*
