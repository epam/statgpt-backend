import pytest
from pydantic import ValidationError

from statgpt.common.schemas.data_query_tool import DataQueryDetails, DataQueryStageNames
from statgpt.common.schemas.tool_details import StageDescriptor, StageRules, StagesConfig


class TestDataQueryStageNamesDefaults:
    """The default display names must match the strings used historically as hardcoded
    literals so that channels that don't configure overrides keep their current display names.
    Each stage descriptor's `key` must equal the field name."""

    @pytest.mark.parametrize(
        "field, expected",
        [
            ("normalizing_query", "Normalizing Query"),
            ("extracting_named_entities", "Extracting Named Entities"),
            ("hybrid_indicators_selection", "Hybrid Indicators Selection"),
            ("selecting_indicators", "Selecting Indicators"),
            ("selecting_special_dimensions", "Selecting Special Dimensions"),
            ("constructing_data_query", "Constructing Data Query"),
            ("executing_data_query", "Executing Data Query"),
        ],
    )
    def test_default_descriptor_matches_legacy_string(self, field: str, expected: str):
        stage_names = DataQueryStageNames()
        descriptor: StageDescriptor = getattr(stage_names, field)
        assert descriptor.key == field
        assert descriptor.name == expected

    def test_data_query_details_exposes_pipeline_stage_names(self):
        details = DataQueryDetails()
        assert isinstance(details.pipeline_stage_names, DataQueryStageNames)
        assert details.pipeline_stage_names.normalizing_query.name == "Normalizing Query"
        assert details.pipeline_stage_names.normalizing_query.key == "normalizing_query"

    def test_yaml_string_override_roundtrip_snake_case(self):
        """YAML can override a stage by writing a plain string for its display name,
        using the snake_case field names directly (allowed by `populate_by_name=True`).
        """
        details = DataQueryDetails.model_validate(
            {"pipeline_stage_names": {"normalizing_query": "Preparing your query"}}
        )
        descriptor = details.pipeline_stage_names.normalizing_query
        assert descriptor.key == "normalizing_query"
        assert descriptor.name == "Preparing your query"
        # Non-overridden fields keep their defaults
        assert (
            details.pipeline_stage_names.extracting_named_entities.name
            == "Extracting Named Entities"
        )

    def test_yaml_string_override_roundtrip_camel_case(self):
        """The string shorthand must also work when YAML uses the camelCase aliases,
        since `BaseYamlModel` configures `alias_generator=to_camel` and channel YAMLs
        are written in camelCase (this is the form shown in the docstring example).
        """
        details = DataQueryDetails.model_validate(
            {"pipelineStageNames": {"normalizingQuery": "Preparing your query"}}
        )
        descriptor = details.pipeline_stage_names.normalizing_query
        assert descriptor.key == "normalizing_query"
        assert descriptor.name == "Preparing your query"
        assert (
            details.pipeline_stage_names.extracting_named_entities.name
            == "Extracting Named Entities"
        )

    def test_user_supplied_key_is_ignored(self):
        """Even if YAML tries to set `key`, it is locked to the field name."""
        details = DataQueryDetails.model_validate(
            {
                "pipeline_stage_names": {
                    "normalizing_query": {"key": "bogus_key", "name": "Foo"},
                }
            }
        )
        descriptor = details.pipeline_stage_names.normalizing_query
        assert descriptor.key == "normalizing_query"
        assert descriptor.name == "Foo"

    def test_descriptor_is_debug_delegates_to_stages_config(self):
        details = DataQueryDetails.model_validate(
            {
                "stages_config": {
                    "debug_only": True,
                    "rules": [{"key": "normalizing_query", "debug_only": False}],
                }
            }
        )
        stage_names = details.pipeline_stage_names
        assert stage_names.normalizing_query.is_debug(details.stages_config) is False
        assert stage_names.executing_data_query.is_debug(details.stages_config) is True


class TestStageRulesValidation:
    def test_pattern_only_is_valid(self):
        rule = StageRules(pattern="^Normalizing", debug_only=False)
        assert rule.pattern == "^Normalizing"
        assert rule.key is None

    def test_key_only_is_valid(self):
        rule = StageRules(key="normalizing_query", debug_only=False)
        assert rule.key == "normalizing_query"
        assert rule.pattern is None

    def test_both_pattern_and_key_raises(self):
        with pytest.raises(ValidationError):
            StageRules(pattern="^Normalizing", key="normalizing_query", debug_only=False)

    def test_neither_pattern_nor_key_raises(self):
        with pytest.raises(ValidationError):
            StageRules(debug_only=False)


class TestStagesConfigIsStageDebug:
    def test_default_when_no_rules(self):
        config = StagesConfig(debug_only=True)
        assert config.is_stage_debug(key="normalizing_query", name="Normalizing Query") is True

        config = StagesConfig(debug_only=False)
        assert config.is_stage_debug(key="normalizing_query", name="Normalizing Query") is False

    def test_key_rule_matches_by_key(self):
        config = StagesConfig(
            debug_only=True,
            rules=[StageRules(key="normalizing_query", debug_only=False)],
        )
        assert config.is_stage_debug(key="normalizing_query", name="Anything") is False

    def test_key_rule_does_not_match_when_key_mismatches(self):
        config = StagesConfig(
            debug_only=True,
            rules=[StageRules(key="normalizing_query", debug_only=False)],
        )
        assert config.is_stage_debug(key="other_key", name="Normalizing Query") is True

    def test_pattern_rule_matches_display_name(self):
        config = StagesConfig(
            debug_only=True,
            rules=[StageRules(pattern="^Normalizing", debug_only=False)],
        )
        assert config.is_stage_debug(key="normalizing_query", name="Normalizing Query") is False

    def test_pattern_rule_ignores_key(self):
        """A pattern rule should not coincidentally match because key string
        happens to match the regex."""
        config = StagesConfig(
            debug_only=True,
            rules=[StageRules(pattern="^normalizing_query$", debug_only=False)],
        )
        # name is the display name, which doesn't match the regex
        assert config.is_stage_debug(key="normalizing_query", name="Normalizing Query") is True

    def test_first_matching_rule_wins(self):
        config = StagesConfig(
            debug_only=True,
            rules=[
                StageRules(key="normalizing_query", debug_only=False),
                StageRules(pattern=".*", debug_only=True),
            ],
        )
        assert config.is_stage_debug(key="normalizing_query", name="Normalizing Query") is False

    def test_backward_compat_pattern_only_yaml(self):
        """Legacy YAMLs that only define `pattern:` continue to work."""
        config = StagesConfig.model_validate(
            {
                "debug_only": True,
                "rules": [{"pattern": "^Constructing", "debug_only": False}],
            }
        )
        assert config.is_stage_debug(name="Constructing Data Query") is False
        assert config.is_stage_debug(name="Other Stage") is True
