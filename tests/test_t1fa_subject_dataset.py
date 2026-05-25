import torch


def test_subject_slice_dataset_returns_image_pair_and_metadata():
    from src.data.subject_index import load_subject_index
    from src.data.t1fa_subject_dataset import T1FASubjectSliceDataset

    subject_index = load_subject_index(
        excel_path="data/data_information.xlsx",
        split_json="data/processed/dataset_splits.json",
    )
    dataset = T1FASubjectSliceDataset(
        t1_dir="data/processed/test/t1_slices",
        fa_dir="data/processed/test/fa_slices",
        subject_index=subject_index,
    )

    sample = dataset[0]

    assert len(dataset) == 1900
    assert sample["t1_slice"].shape == torch.Size([3, 224, 224])
    assert sample["fa_slice"].shape == torch.Size([3, 224, 224])
    assert sample["t1_slice"].min().item() >= -1.0
    assert sample["t1_slice"].max().item() <= 1.0
    assert sample["fa_slice"].min().item() >= -1.0
    assert sample["fa_slice"].max().item() <= 1.0
    assert sample["fname"].endswith(".png")
    assert sample["subject_id"].startswith("sub-")
    assert isinstance(sample["slice_id"], int)
    assert sample["group_name"] in {"CN", "SCD", "MCI", "AD"}
    assert sample["split"] == "test"

