# Limitations

- The reported 0.288729 Dice is a validation result after iterative method
  development. The one locked final test evaluation obtained Dice 0.260881;
  later validation ablations cannot convert that already-opened test into a new
  unseen test for a changed pipeline.
- Test localization assumes that the binary tumor/normal image label is
  available. This differs from label-free semantic-segmentation inference.
- The `<1%` subgroup remains difficult: 35/94 validation cases are complete
  misses and selected masks are heavily over-sized.
- The `>=5%` subgroup contains only 18 validation images, so its mean is
  uncertain and selected masks are generally too small.
- The candidate gallery depends on large frozen foundation models and SAM,
  increasing offline compute and storage.
- BTXRD does not publish verified patient identifiers; canonical grouping is a
  documented filename/metadata heuristic.
- Pixel spacing is unavailable, so area groups and overlap metrics operate in
  image pixels rather than physical units.
