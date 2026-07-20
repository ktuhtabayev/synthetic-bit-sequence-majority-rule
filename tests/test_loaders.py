from __future__ import annotations

from pathlib import Path

import numpy as np

from synthetic_bit_sequence_majority_rule.io.loaders import load_dataset_from_path


def test_default_csv_and_dat_load_with_same_core_data() -> None:
    project_root = Path(__file__).resolve().parents[1]
    csv_dataset = load_dataset_from_path(project_root / "datasets" / "default.csv")
    dat_dataset = load_dataset_from_path(project_root / "datasets" / "default.dat")

    assert csv_dataset.n_objects == dat_dataset.n_objects == 10
    assert csv_dataset.n_features == dat_dataset.n_features == 6
    assert csv_dataset.class_counts == dat_dataset.class_counts == {1: 6, 2: 4}
    assert csv_dataset.object_labels == dat_dataset.object_labels
    assert csv_dataset.feature_names == dat_dataset.feature_names
    assert np.allclose(csv_dataset.X, dat_dataset.X)
    assert np.array_equal(csv_dataset.y, dat_dataset.y)


def test_default_dataset_metadata_describes_two_class_quantitative_data() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset = load_dataset_from_path(project_root / "datasets" / "default.dat")

    assert dataset.metadata["m_objects"] == 10
    assert dataset.metadata["n_features"] == 6
    assert dataset.metadata["n_classes"] == 2
    assert dataset.metadata["feature_signs"] == [1, 1, 1, 1, 1, 1]
    assert dataset.metadata["all_features_quantitative"] is True


def test_ionosfera_csv_and_dat_load_with_same_core_data() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root / "datasets" / "quantitative" / "ionosfera"
    csv_dataset = load_dataset_from_path(dataset_dir / "Ionosfera (350, 33, 2).csv")
    dat_dataset = load_dataset_from_path(dataset_dir / "Ionosfera (350, 33, 2).dat")

    assert csv_dataset.n_objects == dat_dataset.n_objects == 350
    assert csv_dataset.n_features == dat_dataset.n_features == 33
    assert csv_dataset.class_counts == dat_dataset.class_counts == {1: 125, 2: 225}
    assert csv_dataset.object_labels == dat_dataset.object_labels
    assert csv_dataset.feature_names == dat_dataset.feature_names
    assert np.allclose(csv_dataset.X, dat_dataset.X)
    assert np.array_equal(csv_dataset.y, dat_dataset.y)


def test_headerless_csv_without_shape_row_respects_has_header_false(tmp_path: Path) -> None:
    path = tmp_path / "headerless.csv"
    path.write_text(
        "1.0,2.0,3.0,1\n"
        "2.0,3.0,4.0,1\n"
        "8.0,9.0,7.0,2\n"
        "9.0,8.0,6.0,2\n",
        encoding="utf-8",
    )

    dataset = load_dataset_from_path(path, has_header=False)

    assert dataset.n_objects == 4
    assert dataset.n_features == 3
    assert dataset.class_counts == {1: 2, 2: 2}
    assert np.allclose(dataset.X[0], [1.0, 2.0, 3.0])


def test_ionosfera_metadata_treats_all_features_as_quantitative() -> None:
    project_root = Path(__file__).resolve().parents[1]
    path = (
        project_root
        / "datasets"
        / "quantitative"
        / "ionosfera"
        / "Ionosfera (350, 33, 2).csv"
    )
    dataset = load_dataset_from_path(path)

    assert dataset.metadata["m_objects"] == 350
    assert dataset.metadata["n_features"] == 33
    assert dataset.metadata["n_classes"] == 2
    assert dataset.metadata["feature_signs"] == [1] * 33
    assert dataset.metadata["quantitative_feature_indices"] == list(range(33))
    assert dataset.metadata["nominal_feature_indices"] == []
    assert dataset.metadata["all_features_quantitative"] is True
    assert dataset.metadata["has_nominal_features"] is False
