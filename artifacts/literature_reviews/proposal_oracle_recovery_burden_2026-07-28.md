# Proposal-oracle recovery burden for the BTXRD mask-bag branch

Date: 2026-07-28  
Status: arithmetic audit of frozen evidence; no new prediction or GT access

## Inputs

All values come from already frozen protocols:

- current deployable proposal selector and candidate oracle:
  `rad_dino_mask_bag_mil_probe_val_v1`;
- operational goals:
  `wsss_feasible_validation_goal_v1`;
- conditional consumer-entry tier:
  `post_prediction_consumer_entry_gate_v1`.

No per-image prediction, validation mask or test record was opened for this
calculation.

## Burden calculation

For each subgroup:

`gap recovery = (threshold - current) / (oracle - current)`.

| Group | Current selector | Candidate oracle | Entry tier | Goal | Entry gap recovery | Goal gap recovery | Goal/oracle | Oracle headroom above goal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 0.23366822 | 0.40907629 | 0.28513013 | 0.34024039 | 29.34% | 60.76% | 83.17% | 0.06883590 |
| small | 0.11152529 | 0.22274968 | 0.12497817 | 0.17895493 | 12.10% | 60.62% | 80.34% | 0.04379475 |
| medium | 0.34768577 | 0.59414817 | 0.46292241 | 0.51244178 | 46.76% | 66.85% | 86.25% | 0.08170639 |
| large | 0.41545552 | 0.64182777 | 0.41031012 | 0.49370336 | -2.27% | 34.57% | 76.92% | 0.14812441 |

Large already exceeds the entry tier under this selector reference. Small has
the narrowest absolute oracle headroom, but medium imposes the hardest
selection-efficiency requirement: a WTA selector would need to recover about
two-thirds of its current-to-oracle gap and reach over 86% of oracle mean Dice.

## Implications

1. **Correction v3 is still worthwhile.** It removes a widespread spatial
   descriptor defect without changing proposals or supervision.
2. **Direct operational-goal success is a high bar for WTA.** Small is not the
   only difficult subgroup; a mechanism optimized only for tiny masks can
   still fail medium.
3. **The consumer-entry tier is appropriately staged.** It asks for only
   12.10% small-gap recovery, but 46.76% for medium, preventing a small-only
   improvement from authorizing consumer training.
4. **Oracle pass is not deployable evidence.** It only proves candidate
   support. Image-label MIL must select candidates without access to the
   oracle.
5. **Relational selection is conditionally justified.** If corrected
   independent scoring passes oracle but not entry, family-normalized
   cross-fitted relational MIL targets selection efficiency. If it passes
   entry but not goals, the separately gated robust consumer may combine
   spatial evidence rather than forcing one proposal to achieve near-oracle
   Dice.
6. **No subgroup-specific training route is allowed.** The burden analysis is
   post-freeze evaluation arithmetic; subgroup identity remains forbidden for
   training, loss weighting or routing.

## Literature context

Correct WSOL protocol separation and the danger of using localization GT for
model selection:

- Choe et al., *Evaluating Weakly Supervised Object Localization Methods
  Right*, CVPR 2020:
  https://openaccess.thecvf.com/content_CVPR_2020/html/Choe_Evaluating_Weakly_Supervised_Object_Localization_Methods_Right_CVPR_2020_paper.html

Proposal-aligned MIL over arbitrary mask shapes:

- Shen et al., *Toward Joint Thing-and-Stuff Mining for Weakly Supervised
  Panoptic Segmentation*, CVPR 2021:
  https://openaccess.thecvf.com/content/CVPR2021/html/Shen_Toward_Joint_Thing-and-Stuff_Mining_for_Weakly_Supervised_Panoptic_Segmentation_CVPR_2021_paper.html

Relational critical-instance MIL:

- Li, Li and Eliceiri, *Dual-Stream Multiple Instance Learning Network for
  Whole Slide Image Classification with Self-supervised Contrastive
  Learning*, CVPR 2021:
  https://openaccess.thecvf.com/content/CVPR2021/html/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_With_Self-Supervised_CVPR_2021_paper.html

These sources motivate protocol and mechanism choices. Their reported metrics
are not converted into BTXRD Dice.
