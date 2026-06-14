import pandas as pd
import numpy as np

from scripts.evaluate_finalpdf_train_test_roi import (
    MethodPair,
    build_fused_feature_table_local,
    build_fused_slice_feature_table_local,
    evaluate_train_test_classifier,
    extract_slice_features_with_atlas,
    parse_method_pair,
)
from scripts.make_atlas_slice_masks import resize_and_pad_mask


def test_parse_method_pair_accepts_train_and_test_dirs():
    pair = parse_method_pair("Ours=outputs/train_preds:outputs/test_preds")

    assert pair == MethodPair("Ours", "outputs/train_preds", "outputs/test_preds")


def test_train_test_classifier_reports_high_acc_auc_for_separable_roi_features():
    train_rows = []
    test_rows = []
    for idx in range(8):
        train_rows.append(
            {
                "method": "Ours",
                "subject_id": f"train-cn-{idx}",
                "group_name": "CN",
                "roi_label_1_mean": 0.1 + idx * 0.01,
                "roi_label_2_mean": 0.2,
            }
        )
        train_rows.append(
            {
                "method": "Ours",
                "subject_id": f"train-mci-{idx}",
                "group_name": "MCI",
                "roi_label_1_mean": 0.8 - idx * 0.01,
                "roi_label_2_mean": 0.7,
            }
        )
    for idx in range(4):
        test_rows.append(
            {
                "method": "Ours",
                "subject_id": f"test-cn-{idx}",
                "group_name": "CN",
                "roi_label_1_mean": 0.12 + idx * 0.01,
                "roi_label_2_mean": 0.2,
            }
        )
        test_rows.append(
            {
                "method": "Ours",
                "subject_id": f"test-mci-{idx}",
                "group_name": "MCI",
                "roi_label_1_mean": 0.78 - idx * 0.01,
                "roi_label_2_mean": 0.7,
            }
        )

    result, predictions = evaluate_train_test_classifier(
        pd.DataFrame(train_rows),
        pd.DataFrame(test_rows),
        method="Ours",
        task="cn_vs_mci",
        classifier="linear_svm",
        feature_set="roi_mean",
        max_features=1,
    )

    assert result["accuracy"] == 1.0
    assert result["macro_auc_ovr"] == 1.0
    assert result["n_train_subjects"] == 16
    assert result["n_test_subjects"] == 8
    assert len(predictions) == 8


def test_fusion_keeps_atlas_roi_label_mean_features():
    features = pd.DataFrame(
        [
            {"method": "T1", "subject_id": "sub-001", "group_name": "CN", "split": "train", "roi_label_1_mean": 0.1},
            {"method": "FA", "subject_id": "sub-001", "group_name": "CN", "split": "train", "roi_label_1_mean": 0.2},
        ]
    )

    fused = build_fused_feature_table_local(features, "T1_PLUS_FA", "T1", "FA", feature_set="roi_mean")

    assert fused.loc[0, "method"] == "T1_PLUS_FA"
    assert fused.loc[0, "T1__roi_label_1_mean"] == 0.1
    assert fused.loc[0, "FA__roi_label_1_mean"] == 0.2


def test_slice_atlas_features_keep_each_slice_as_one_sample(tmp_path):
    import cv2

    image_dir = tmp_path / "images"
    atlas_dir = tmp_path / "atlas"
    image_dir.mkdir()
    atlas_dir.mkdir()
    for z, value in [(20, 51), (21, 153)]:
        image = np.full((4, 4), value, dtype=np.uint8)
        mask = np.ones((4, 4), dtype=np.uint8)
        cv2.imwrite(str(image_dir / f"sub-001_z{z:03d}.png"), image)
        cv2.imwrite(str(atlas_dir / f"atlas_z{z:03d}.png"), mask)

    subject_index = pd.DataFrame(
        [{"subject_id": "sub-001", "group_id": 1, "group_name": "CN", "split": "train"}]
    )

    features = extract_slice_features_with_atlas(
        image_dir=image_dir,
        atlas_dir=atlas_dir,
        method="FA",
        subject_index=subject_index,
        split="train",
        min_pixels=1,
    )

    assert features["sample_id"].tolist() == ["sub-001_z020", "sub-001_z021"]
    assert features["slice_idx"].tolist() == [20, 21]
    assert features["roi_label_1_mean"].round(3).tolist() == [0.2, 0.6]


def test_slice_fusion_aligns_on_subject_and_slice():
    features = pd.DataFrame(
        [
            {"method": "T1", "sample_id": "sub-001_z020", "subject_id": "sub-001", "slice_idx": 20, "group_name": "CN", "split": "train", "roi_label_1_mean": 0.1},
            {"method": "T1", "sample_id": "sub-001_z021", "subject_id": "sub-001", "slice_idx": 21, "group_name": "CN", "split": "train", "roi_label_1_mean": 0.2},
            {"method": "FA", "sample_id": "sub-001_z020", "subject_id": "sub-001", "slice_idx": 20, "group_name": "CN", "split": "train", "roi_label_1_mean": 0.3},
        ]
    )

    fused = build_fused_slice_feature_table_local(features, "T1_PLUS_FA", "T1", "FA", feature_set="roi_mean")

    assert fused["sample_id"].tolist() == ["sub-001_z020"]
    assert fused.loc[0, "T1__roi_label_1_mean"] == 0.1
    assert fused.loc[0, "FA__roi_label_1_mean"] == 0.3


def test_resize_and_pad_mask_preserves_discrete_labels():
    mask = np.zeros((2, 4), dtype=np.int32)
    mask[:, 1:3] = 7

    out = resize_and_pad_mask(mask, target_size=8)

    assert out.shape == (8, 8)
    assert set(np.unique(out).tolist()) == {0, 7}
