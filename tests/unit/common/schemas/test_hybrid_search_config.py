import warnings

import pytest

from statgpt.common.schemas.data_query_tool import HybridSearchConfig


class TestIncludeLowerScoredDatasetsBackCompat:
    """`use_only_best_score` was renamed to `include_lower_scored_datasets` with
    inverse meaning. The old field is kept as a deprecated back-compat alias."""

    def test_default(self):
        cfg = HybridSearchConfig.model_validate({})
        assert cfg.include_lower_scored_datasets is False
        with pytest.warns(DeprecationWarning):
            assert cfg.use_only_best_score is None

    def test_validation_does_not_warn(self):
        """The back-compat validator reads the deprecated field internally without
        triggering its DeprecationWarning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            HybridSearchConfig.model_validate({"useOnlyBestScore": True})

    def test_new_field_used_as_is(self):
        cfg = HybridSearchConfig.model_validate({"includeLowerScoredDatasets": True})
        assert cfg.include_lower_scored_datasets is True

    def test_deprecated_field_maps_when_new_absent(self):
        # useOnlyBestScore=False means "include lower scored datasets"
        cfg = HybridSearchConfig.model_validate({"useOnlyBestScore": False})
        assert cfg.include_lower_scored_datasets is True

        cfg = HybridSearchConfig.model_validate({"useOnlyBestScore": True})
        assert cfg.include_lower_scored_datasets is False

    def test_new_field_wins_when_both_set(self):
        cfg = HybridSearchConfig.model_validate(
            {"includeLowerScoredDatasets": False, "useOnlyBestScore": False}
        )
        assert cfg.include_lower_scored_datasets is False

    def test_deprecated_field_accepts_snake_case(self):
        cfg = HybridSearchConfig.model_validate({"use_only_best_score": False})
        assert cfg.include_lower_scored_datasets is True
