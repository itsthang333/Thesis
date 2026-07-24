# External checkpoints

Model files are intentionally excluded from Git because they are hundreds of
megabytes. Resolve every file by the SHA-256 and byte count in
`checkpoint_pointer.json`; a filename alone is not provenance.

The official thesis result uses only the WSSS segmenter checkpoint whose hash
starts with `02d3af8f`. The fully supervised checkpoint whose hash starts with
`05606a0a` is a separate upper-bound diagnostic and is never an alternative
official checkpoint.
