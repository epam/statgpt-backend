# sdmx-json-data-parsing Specification

## Purpose

Defines how SDMX-JSON data responses are read into series, observations and attribute values. This covers SDMX-JSON 1.0, returned by SDMX 2.1 APIs (QuantHub included), and SDMX-JSON 2.0, returned by the StatGPT SDMX proxy and other SDMX 3.0 APIs. The goal is that every series carries the attribute values the data source sent for it.

## Requirements

### Requirement: StatGPT's own reader reads SDMX-JSON
Every SDMX-JSON data response SHALL be read by StatGPT's own reader, never by the sdmx1 library's JSON reader. The sdmx1 reader resolves series attributes from the `dataSet` attribute array, never attaches `dataSet` attributes, and drops attributes without values before matching positions. As a result, series lose their attributes or all get the same value.

#### Scenario: SDMX 2.1 source requesting JSON
- **WHEN** an SDMX 2.1 data source with `Text/JSON` as its data `accept` header returns a data message
- **THEN** it is read by StatGPT's reader

### Requirement: Both SDMX-JSON layouts are accepted
The reader SHALL accept the SDMX-JSON 1.0 layout (top-level `header`, `structure` and `dataSets`) and the 2.0 layout (`meta`, with `structures` and `dataSets` under `data`), and SHALL treat the level name `dataset` as `dataSet`.

#### Scenario: Same data in both layouts
- **WHEN** the same series arrive once as SDMX-JSON 1.0 and once as SDMX-JSON 2.0
- **THEN** both are read into the same series, observations and attribute values

### Requirement: Series attributes come from the series' own array
The attributes of a series SHALL be resolved from that series' own attribute index array, matched by position against the `series` attributes of the structure. A `null` position means the attribute does not apply to the series.

#### Scenario: Two series with different scales
- **WHEN** the structure lists the series attributes `DECIMALS_DISPLAYED` (value `3`), `FUNCTIONAL_CAT` (no values) and `SCALE` (values `9` and `0`), series `0:0` has the array `[0, null, 0]` and series `0:1` has `[0, null, 1]`
- **THEN** series `0:0` has `DECIMALS_DISPLAYED` = `3` and `SCALE` = `9`, and series `0:1` has `DECIMALS_DISPLAYED` = `3` and `SCALE` = `0`

### Requirement: Attributes without values keep their position
An attribute that the structure lists with no values SHALL keep its position, so the attributes after it are still matched to the right array element.

#### Scenario: Empty attribute between two coded ones
- **WHEN** `FUNCTIONAL_CAT`, listed second among the series attributes, has no values and a series array is `[0, null, 1]`
- **THEN** the third element resolves `SCALE`, not the attribute after it

### Requirement: Dataset attributes are attached to the dataset
The `attributes` array of a data set SHALL be resolved against the structure's `dataSet` attributes and attached to the data set. When the array omits trailing positions, an attribute with exactly one coded value SHALL take that value.

#### Scenario: Publisher of the data set
- **WHEN** the structure lists the `dataSet` attribute `PUBLISHER` with the single value `IMF` and the data set's `attributes` is `[0]`
- **THEN** the data set carries `PUBLISHER` = `IMF`

### Requirement: Single-valued attributes that no series references apply to every series
A `series` attribute that no series of the data set references, and that has exactly one value in the structure, SHALL be applied to every series. Some servers list the values of attributes that depend on a subset of the series dimensions (such as `UNIT` or `BASE_YEAR`) in the structure but leave `null` at their position in every series. A single value can only belong to every series in the message. An unreferenced attribute with two or more values SHALL NOT be applied.

#### Scenario: Unit listed only in the structure
- **WHEN** the series attribute `UNIT` has the single value `USD` and every series has `null` at its position
- **THEN** every series carries `UNIT` = `USD`

#### Scenario: Ambiguous unreferenced attribute
- **WHEN** the series attribute `UNIT` has the values `USD` and `EUR` and every series has `null` at its position
- **THEN** no series carries `UNIT`

### Requirement: Group attributes apply to the series they match
Attributes of the SDMX-JSON 2.0 `dimensionGroup` level SHALL be read from the data set's `dimensionGroupAttributes`, which are keyed by a partial series key, where an empty segment matches any value. Each group SHALL be applied to every series whose key matches the group's key in every constrained dimension. These attributes are not part of the series attribute arrays.

#### Scenario: Attribute per indicator
- **WHEN** the series dimensions are `COUNTRY` and `INDICATOR`, and `dimensionGroupAttributes` holds `":0"` with the note "Nominal GDP"
- **THEN** every series of the first `INDICATOR` value carries that note, whatever its `COUNTRY`

### Requirement: Uncoded values are kept as text
An attribute value given as a string instead of an index SHALL be kept as that string. An inline list of values SHALL be joined with `, `, taking the first available localization of each localized text. A dimension value given as `value` instead of `id` SHALL be read the same way as an `id`.

#### Scenario: Localized inline value
- **WHEN** a group attribute's value is the inline list `[{"en": "Nominal GDP"}]`
- **THEN** the matching series carry the attribute value "Nominal GDP"
