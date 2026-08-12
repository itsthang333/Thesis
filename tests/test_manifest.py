from __future__ import annotations

from btxrd_wsss.data.manifest import read_manifest


def test_reads_frozen_audit_schema_and_assigns_group_stable_train_fold(tmp_path):
    manifest = tmp_path / "frozen.csv"
    manifest.write_text(
        "image_id,group_id,split,tumor,tumor_type,anatomy,view\n"
        "a.png,case-1,train,1,9,femur,frontal\n"
        "b.png,case-1,train,1,9,femur,lateral\n"
        "c.png,case-2,val,0,0,tibia,frontal\n",
        encoding="utf-8",
    )

    records = read_manifest(manifest, data_root=tmp_path)

    assert records[0].class_index == 9
    assert records[0].class_indices == (9,)
    assert records[0].fold is not None
    assert records[0].fold == records[1].fold
    assert records[2].fold is None
