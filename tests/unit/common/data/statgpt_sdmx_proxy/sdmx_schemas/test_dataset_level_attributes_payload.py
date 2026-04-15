from statgpt.common.data.statgpt_sdmx_proxy.sdmx_schemas.dataset_level_attributes_payload import (
    parse_dataset_level_attributes_map,
)


def test_parse_proxy_style_lowercase_dataset_key_and_indexed_values() -> None:
    payload = {
        "data": {
            "dataSets": [{"attributes": [0, 1]}],
            "structures": [
                {
                    "attributes": {
                        "dataset": [
                            {
                                "id": "ATTR_A",
                                "values": [{"id": "CODE_A", "name": "Label A"}],
                            },
                            {
                                "id": "ATTR_B",
                                "values": [
                                    {"id": "CODE_B", "name": "Label B"},
                                    {"id": "CODE_B2", "name": "Other"},
                                ],
                            },
                        ],
                        "observation": [],
                    },
                }
            ],
        }
    }
    assert parse_dataset_level_attributes_map(payload) == {
        "ATTR_A": "CODE_A",
        "ATTR_B": "CODE_B2",
    }


def test_parse_registry_style_data_set_key_and_inline_lists() -> None:
    payload = {
        "data": {
            "dataSets": [
                {
                    "attributes": [
                        None,
                        [{"en": "Synthetic description text."}],
                        0,
                    ]
                }
            ],
            "structures": [
                {
                    "attributes": {
                        "dataSet": [
                            {"id": "NO_VALUE", "values": []},
                            {
                                "id": "FULL_DESCRIPTION",
                                "values": [],
                            },
                            {
                                "id": "PUBLISHER",
                                "values": [{"id": "AG1"}],
                            },
                        ],
                    },
                }
            ],
        }
    }
    assert parse_dataset_level_attributes_map(payload) == {
        "NO_VALUE": None,
        "FULL_DESCRIPTION": "{'en': 'Synthetic description text.'}",
        "PUBLISHER": "AG1",
    }


def test_structure_on_payload_root() -> None:
    payload = {
        "data": {"dataSets": [{"attributes": [0]}], "structures": []},
        "structure": {
            "attributes": {
                "dataset": [
                    {"id": "ONLY", "values": [{"id": "X", "name": "XN"}]},
                ],
            },
        },
    }
    assert parse_dataset_level_attributes_map(payload) == {"ONLY": "X"}


def test_infer_missing_trailing_index_single_value() -> None:
    payload = {
        "data": {
            "dataSets": [{"attributes": []}],
            "structures": [
                {
                    "attributes": {
                        "dataSet": [
                            {"id": "INFERRED", "values": [{"id": "SOLE"}]},
                        ],
                    },
                }
            ],
        }
    }
    assert parse_dataset_level_attributes_map(payload) == {"INFERRED": "SOLE"}


def test_invalid_payload_returns_empty() -> None:
    assert parse_dataset_level_attributes_map({}) == {}
    assert parse_dataset_level_attributes_map({"data": {"dataSets": [], "structures": []}}) == {}
