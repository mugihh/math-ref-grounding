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
