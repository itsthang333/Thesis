# Rich-gallery oracle feature-gap diagnostic

## Scope

This is a post-freeze validation diagnostic. It compares the immutable
`0.5*rank(G1)+0.5*rank(upstream)` choice with the already-evaluated gallery
oracle. It does not train a selector, tune a threshold, or open test data.

Tumor images: `184`; immutable baseline Dice: `0.288729487`; gallery oracle
Dice: `0.528298332`.

## Rank evidence: existing scores suppress the oracle

| Signal | Oracle lower-ranked fraction | Median oracle-selected rank delta |
|---|---:|---:|
| `g1_rank` | 0.842 | -0.1621 |
| `upstream_rank` | 0.870 | -0.2167 |
| `sam_rank` | 0.707 | -0.2291 |

A global weight sweep cannot recover a candidate that is simultaneously
ranked below the frozen choice by the available signals. The failure is
missing candidate-level tumor identity/extent evidence, not coefficient choice.

## Extent direction changes with lesion size

| Group | n | Median oracle-selected area | Oracle smaller | Oracle larger |
|---|---:|---:|---:|---:|
| `small` | 94 | -0.001299 | 0.574 | 0.394 |
| `medium` | 72 | +0.001538 | 0.431 | 0.528 |
| `large` | 18 | +0.060391 | 0.167 | 0.833 |

The required correction reverses between tiny and large lesions. Therefore
raw area, dilation, consensus expansion, or one global morphology rule is
structurally incapable of improving all subgroups together.

## Source mismatch

| Group | Selected external | Oracle external | Selected classifier | Oracle classifier |
|---|---:|---:|---:|---:|
| `small` | 0.106 | 0.245 | 0.468 | 0.426 |
| `medium` | 0.125 | 0.417 | 0.500 | 0.250 |
| `large` | 0.278 | 0.556 | 0.444 | 0.056 |

The oracle increasingly shifts to external-saliency candidates as lesion size
grows, while the frozen selector keeps over-selecting classifier-derived masks.
A source router without a reliable lesion-scale/tumor-evidence variable cannot
know when to make that switch; this explains why G2 routing did not beat the
fixed fusion.

## Additional shape and prompt evidence

- The oracle has lower prompt inside-versus-ring contrast in `64.7%` of cases.
  Therefore another score derived monotonically from the same prompt heatmap
  is likely to reinforce the wrong discriminative fragment rather than reveal
  complete tumor extent.
- The oracle is less compact in `55.4%` of cases and contains more connected
  components in `50.0%`. The selector's preference for compact, easy masks is
  not aligned with the heterogeneous morphology of bone tumors.
- No available geometry scalar has a stable overall direction exceeding the
  rank failures above. Geometry can condition a genuinely tumor-specific score,
  but it cannot replace that score.

## Research decision

1. Keep the rich gallery and the `0.288729` baseline immutable.
2. Do not run another G1/upstream/SAM/area/source-weight sweep.
3. Require a new candidate-conditioned positive-evidence score whose direction
   is not inherited from those three ranks and whose extent response is
   conditioned by image evidence rather than raw mask area.
4. The BAS-Softplus probe tests exactly whether inside-versus-background
   activation can supply that missing variable. It must pass mechanics gates
   before any full run and must ultimately beat actual Dice `0.288729`.

Validation GT-derived oracle information in this dossier is diagnostic only.
It cannot authorize a learned selector or count as a deployment result.
