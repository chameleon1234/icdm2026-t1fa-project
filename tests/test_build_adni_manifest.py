import pandas as pd
import pytest


def test_normalize_adni_subject_id_from_file_and_csv():
    from scripts.build_adni_manifest import normalize_adni_subject_id

    assert normalize_adni_subject_id("sub-002S0413_space-MNI152NLin6Asym_res-02_FA_brain.nii.gz") == "002_S_0413"
    assert normalize_adni_subject_id("002_S_0413") == "002_S_0413"
    assert normalize_adni_subject_id("sub-011S10026") == "011_S_10026"


def test_normalize_group_policy():
    from scripts.build_adni_manifest import normalize_group

    assert normalize_group("CN") == "CN"
    assert normalize_group("MCI") == "MCI_spectrum"
    assert normalize_group("EMCI") == "MCI_spectrum"
    assert normalize_group("LMCI") == "MCI_spectrum"
    assert normalize_group("AD") == "AD"
    assert normalize_group("_S_MC") == "EXCLUDE"
    assert normalize_group(None) == "UNLABELED"


def test_build_subject_manifest_pairs_and_labels():
    from scripts.build_adni_manifest import build_subject_manifest

    inventory = pd.DataFrame(
        [
            {"subject": "002_S_0413", "modality": "T1", "path": "ADNI_data/T1/sub-002S0413_T1.nii.gz"},
            {"subject": "002_S_0413", "modality": "FA", "path": "ADNI_data/FA/sub-002S0413_FA.nii.gz"},
            {"subject": "002_S_1155", "modality": "T1", "path": "ADNI_data/T1/sub-002S1155_T1.nii.gz"},
            {"subject": "002_S_1155", "modality": "FA", "path": "ADNI_data/FA/sub-002S1155_FA.nii.gz"},
            {"subject": "002_S_1261", "modality": "T1", "path": "ADNI_data/T1/sub-002S1261_T1.nii.gz"},
        ]
    )
    labels = pd.DataFrame(
        [
            {"Subject": "002_S_0413", "Group": "CN"},
            {"Subject": "002_S_1155", "Group": "EMCI"},
        ]
    )

    manifest = build_subject_manifest(inventory, labels)

    paired = manifest[manifest["is_paired"]].sort_values("subject")
    assert paired["subject"].tolist() == ["002_S_0413", "002_S_1155"]
    assert paired["normalized_group"].tolist() == ["CN", "MCI_spectrum"]
    assert manifest.loc[manifest["subject"].eq("002_S_1261"), "is_paired"].item() is False


def test_assign_subject_splits_no_leakage():
    from scripts.build_adni_manifest import assign_subject_splits

    rows = []
    for idx in range(30):
        rows.append({"subject": f"CN_{idx:03d}", "is_paired": True, "split_group": "CN"})
    for idx in range(18):
        rows.append({"subject": f"MCI_{idx:03d}", "is_paired": True, "split_group": "MCI_spectrum"})
    for idx in range(9):
        rows.append({"subject": f"AD_{idx:03d}", "is_paired": True, "split_group": "AD"})
    for idx in range(12):
        rows.append({"subject": f"UNL_{idx:03d}", "is_paired": True, "split_group": "UNLABELED"})
    manifest = pd.DataFrame(rows)

    split_df, split_json = assign_subject_splits(manifest, seed=7, train_ratio=0.7, val_ratio=0.1)

    assert set(split_df["split"]) == {"train", "val", "test"}
    assert split_df["subject"].is_unique
    split_sets = {split: set(subjects) for split, subjects in split_json.items()}
    assert not (split_sets["train"] & split_sets["val"])
    assert not (split_sets["train"] & split_sets["test"])
    assert not (split_sets["val"] & split_sets["test"])
    assert set().union(*split_sets.values()) == set(manifest["subject"])


def test_assign_subject_splits_rejects_unpaired_rows():
    from scripts.build_adni_manifest import assign_subject_splits

    manifest = pd.DataFrame(
        [
            {"subject": "sub-1", "is_paired": True, "split_group": "CN"},
            {"subject": "sub-2", "is_paired": False, "split_group": "CN"},
        ]
    )

    with pytest.raises(ValueError, match="paired"):
        assign_subject_splits(manifest)
