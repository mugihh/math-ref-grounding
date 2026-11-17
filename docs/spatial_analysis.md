# Spatial Analysis

This analysis uses the final prompt-v4 predictions on all 1,842 benchmark
examples. Spatial measures compare the top-ranked human-annotated candidate
with the gold box; they do not evaluate automatic detection. For multi-gold
examples, each measure uses the most favorable gold box, as defined by the
evaluator.

## Full-benchmark results

| Model | Exact R@1 | Dist.@.05 | Dist.@.10 | Pred. center in gold | Gold center in pred. | Pred. overlap | Gold overlap | IoU | Mean dist. | Median dist. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random (seed 0) | .153 | .288 | .469 | .258 | .243 | .256 | .235 | .173 | .140 | .110 |
| Class prior (seed 0) | .165 | .320 | .514 | .272 | .333 | .266 | .326 | .204 | .131 | .096 |
| CLIP | .242 | .388 | .544 | .356 | .352 | .349 | .342 | .279 | .121 | .083 |
| CLIP + LaTeX | .375 | .473 | .617 | .480 | .425 | .476 | .417 | .399 | .100 | .058 |
| Qwen2.5-VL | .289 | .444 | .628 | .405 | .410 | .400 | .401 | .325 | .100 | .065 |
| Qwen2.5-VL + LaTeX | .486 | .593 | .711 | .565 | .631 | .559 | .628 | .529 | .077 | .013 |
| GPT-5.4 | .500 | .630 | .751 | .611 | .599 | .609 | .584 | .539 | .068 | .000 |
| GPT-5.4 + LaTeX | .719 | .795 | .852 | .767 | .811 | .761 | .803 | .748 | .040 | .000 |

Coverage is 100% for every row. Full-benchmark spatial values mix exact
matches with errors. Because every exact match contributes zero center
distance and unit overlap/IoU, these values should not be interpreted as the
geometry of mistakes.

## Exact-error-only analysis

GPT-5.4 + LaTeX makes 518 exact-ID errors. Among these:

- 27.0% have normalized center distance at most .05 and 47.3% at most .10;
- 39.8% have any positive intersection with a gold box;
- only 7.3% have IoU at least .5;
- the mean and median normalized center distances are .141 and .104;
- the prediction center lies inside a gold box in 17.0% of errors, while the
  gold center lies inside the predicted box in 32.8%;
- mean prediction-overlap and gold-overlap fractions are .152 and .299.

The containment asymmetry indicates a tendency for some wrong predictions to
select a region larger than the gold referent. Nevertheless, most exact-ID
errors are not near-equivalent boxes under a conventional IoU threshold.
Spatial relaxation therefore changes the interpretation of a minority of
errors but does not erase the reference-resolution problem.

For comparison, GPT-5.4 without LaTeX makes 921 exact-ID errors: 50.2% are
within distance .10, 36.2% overlap at all, and 2.8% reach IoU .5. Adding
reviewed symbols greatly reduces the number of errors and raises the fraction
of remaining errors that are close overlapping alternatives, but the
wrong-only mean distance is similar (.141 versus .136).

## GPT-5.4 + LaTeX error slices

| Slice | Errors | Dist.@.05 | Dist.@.10 | Pred. center in gold | Gold center in pred. | Any overlap | IoU >= .5 | Mean IoU | Mean dist. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| English | 307 | .329 | .534 | .189 | .352 | .436 | .078 | .114 | .134 |
| Japanese | 211 | .185 | .384 | .142 | .294 | .341 | .066 | .086 | .151 |
| Implicit mention | 238 | .298 | .534 | .244 | .445 | .546 | .118 | .146 | .122 |
| Explicit mention | 280 | .246 | .421 | .107 | .229 | .271 | .036 | .065 | .157 |
| 1--5 candidates | 66 | .121 | .379 | .136 | .227 | .303 | .076 | .068 | .165 |
| 6--10 candidates | 167 | .347 | .527 | .210 | .383 | .485 | .120 | .142 | .121 |
| 11--15 candidates | 202 | .292 | .530 | .188 | .347 | .396 | .045 | .093 | .122 |
| 16+ candidates | 83 | .181 | .301 | .072 | .253 | .301 | .048 | .073 | .207 |
| Gold equation | 34 | .412 | .618 | .471 | .353 | .588 | .206 | .241 | .117 |
| Gold span | 230 | .270 | .465 | .235 | .413 | .483 | .113 | .152 | .140 |
| Gold term | 200 | .195 | .405 | .045 | .250 | .260 | .020 | .032 | .159 |
| Gold figure part | 47 | .489 | .702 | .149 | .234 | .447 | .000 | .052 | .081 |
| Gold figure | 7 | .286 | .429 | .286 | .286 | .286 | .143 | .129 | .164 |

Figure and figure-part slices are small and receive no candidate LaTeX, so
they are descriptive only. Language slices are also confounded with source,
topic, board style, candidate density, and transcript provenance.

## Object-granularity pattern

Among the 518 GPT-5.4 + LaTeX errors, 60.8% predict the same object type as
the first gold object type. The most frequent predicted/gold type pairs are:

| Predicted type | Gold type | Count |
|---|---|---:|
| span | span | 150 |
| term | term | 121 |
| equation | span | 57 |
| span | term | 48 |
| equation | term | 32 |
| figure_part | figure_part | 30 |
| term | span | 22 |
| span | equation | 17 |
| equation | equation | 14 |

Same-type errors show that spatial proximity is not only a hierarchy effect:
models also confuse repeated or neighboring objects of identical granularity.
The directional containment and equation-to-span/term errors nevertheless
support the qualitative finding that selecting a containing region is one
recurring failure mode.

## Reporting recommendation

Keep exact visual-object identity as the primary metric. If space permits,
report the exact-error-only summary in the appendix rather than adding the
full model table to the body. A concise defensible statement is:

> Among GPT-5.4 + LaTeX's 518 exact-ID errors, 47.3% fall within 10% of the
> image diagonal and 39.8% overlap a gold box, but only 7.3% reach IoU .5;
> spatially close alternatives explain some errors without making exact
> identity evaluation redundant.

