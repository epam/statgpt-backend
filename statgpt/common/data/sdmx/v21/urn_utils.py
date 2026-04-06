from typing import TypeVar

from sdmx.model.common import Version

from .schemas import Urn

T = TypeVar("T")


def is_wildcarded_version(version: Version | str | None) -> bool:
    """Check if a version string contains SDMX wildcard markers ('+')."""
    if isinstance(version, Version):
        return "+" in str(version)
    if isinstance(version, str):
        return "+" in version
    return False


def lookup_urn(artifacts: dict[Urn, T], urn: Urn) -> T:
    """Look up an artifact by URN, resolving wildcarded versions if needed.

    Direct lookup when the version is concrete; falls back to
    ``resolve_wildcarded_urn`` when the version contains '+'.
    """
    if not is_wildcarded_version(urn.version):
        return artifacts[urn]
    return resolve_wildcarded_urn(artifacts, urn)


def resolve_wildcarded_urn(artifacts: dict[Urn, T], wildcarded_urn: Urn) -> T:
    """Find the latest non-wildcarded artifact matching a wildcarded URN.

    Searches for URNs with the same agency_id and resource_id whose
    concrete (non-wildcarded) version satisfies the wildcard pattern,
    then returns the artifact with the highest version.

    Raises KeyError if no matching non-wildcarded URN is found.
    """
    candidates: list[tuple[Urn, T]] = []
    for urn, value in artifacts.items():
        if (
            urn.agency_id == wildcarded_urn.agency_id
            and urn.resource_id == wildcarded_urn.resource_id
            and not is_wildcarded_version(urn.version)
            and _matches_wildcard_version(urn.version, wildcarded_urn.version)
        ):
            candidates.append((urn, value))

    if not candidates:
        raise KeyError(
            f"No non-wildcarded URN matching {wildcarded_urn} found. "
            f"Available URNs: {list(artifacts.keys())}"
        )

    candidates.sort(key=lambda x: _parse_semver_tuple(x[0].version), reverse=True)
    return candidates[0][1]


def _matches_wildcard_version(concrete_version: str, wildcard_version: str) -> bool:
    """Check if a concrete version satisfies a wildcarded version pattern.

    Segments before the first '+' must match exactly.
    From the '+' segment onward the concrete version is compared as >= the
    base value (tuple comparison).
    """
    concrete_semver = concrete_version.split(".")
    wildcard_semver = wildcard_version.split(".")

    wildcard_start = next((i for i, part in enumerate(wildcard_semver) if "+" in part), None)
    if wildcard_start is None:
        return concrete_semver == wildcard_semver

    for i in range(wildcard_start):
        if i >= len(concrete_semver):
            return False
        if concrete_semver[i] != wildcard_semver[i]:
            return False

    min_version = tuple(_semver_part_to_int(p) for p in wildcard_semver[wildcard_start:])
    concrete_suffix = tuple(_semver_part_to_int(p) for p in concrete_semver[wildcard_start:])

    max_len = max(len(min_version), len(concrete_suffix))
    min_version_padded = min_version + (0,) * (max_len - len(min_version))
    concrete_suffix_padded = concrete_suffix + (0,) * (max_len - len(concrete_suffix))

    return concrete_suffix_padded >= min_version_padded


def _semver_part_to_int(part: str) -> int:
    """Extract the numeric value from a semver part (major/minor/patch), ignoring non-digit chars."""
    numeric = "".join(c for c in part if c.isdigit())
    return int(numeric) if numeric else 0


def _parse_semver_tuple(version: str) -> tuple[int, ...]:
    """Parse a semver string (major.minor.patch) into a tuple of integers for comparison."""
    return tuple(_semver_part_to_int(part) for part in version.split("."))
