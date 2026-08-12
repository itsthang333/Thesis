# Vendored model sources

`sam_med2d/` is imported from
[OpenGVLab/SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D) at commit
`bfd2b93b1158100c8abd81f61766a2de92c1c175`.

The upstream code is distributed under the Apache License 2.0; its license is
preserved in `SAM_MED2D_LICENSE`. Local maintenance patches change the
predictor's absolute `segment_anything.modeling` import to the relative package
and load checkpoints in PyTorch's restricted weights-only mode. The namespace
change prevents collisions with original SAM, which remains available as the
legacy backend.
