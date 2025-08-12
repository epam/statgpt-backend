import re
import typing as t
from dataclasses import dataclass


class UrnParseError(Exception):
    def __init__(self, urn: str):
        self.urn = urn

    def __str__(self):
        return f"URN doesn't match pattern: '{self.urn}'"


@dataclass
class Urn:
    agency_id: str
    resource_id: str
    version: str
    resource_type: t.Optional[str] = None
    item_id: str = ""  # empty string means absence

    def get_short_urn(self) -> str:
        urn = f"{self.agency_id}:{self.resource_id}({self.version})"
        if self.item_id:
            urn += f".{self.item_id}"
        return urn

    def get_urn(self) -> str:
        if not self.resource_type:
            raise NoResourceTypeError()
        return f"urn:sdmx:{self.resource_type}={self.get_short_urn()}"

    def to_file_name(self) -> str:
        result = f"{self.agency_id}_{self.resource_id}_{self.version}"
        if self.item_id:
            result += f"_{self.item_id}"
        return result


class UrnParser:
    VERSION_REGEX = r"(?P<version>[\d+]+\.[\d+]+(\.[\d+]+)?(-draft)?)"
    RESOURCE_ID_REGEX = r"(?P<resource_id>[0-9a-zA-Z_\-@\.]+)"
    AGENCY_ID_REGEX = r"(?P<agency_id>[0-9a-zA-Z_\-\.]+)"
    NAMESPACE_REGEX = (
        r"urn:sdmx:[^:]*.(?P<package>[^\.]*)\.(?P<class>[^=]*)"  # todo: improve package path part
    )

    def __init__(self, urn_regex: re.Pattern):
        self.urn_regex = re.compile(urn_regex)

    @classmethod
    def create_default(cls):
        return cls(cls.create_default_urn_regex())

    @classmethod
    def create_default_urn_regex(cls):
        return re.compile(
            rf"(^{cls.NAMESPACE_REGEX}=)?{cls.AGENCY_ID_REGEX}:"
            rf"{cls.RESOURCE_ID_REGEX}\({cls.VERSION_REGEX}\)(\.(?P<item_id>.*))?$"
        )

    def parse(self, urn: t.Optional[str]) -> Urn:
        if urn is None:
            raise UrnParseError("None")
        match = self.urn_regex.match(urn)
        if match is None:
            raise UrnParseError(urn)
        data = match.groupdict()
        res = Urn(
            data["agency_id"],
            data["resource_id"],
            data["version"],
            data["class"],
            data["item_id"] if data["item_id"] else "",
        )
        # logger.info(f'parsed dataflow urn "{urn}" into {res}')
        return res

    def parse_resource(self, urn: t.Union[str, "Urn"]) -> "Urn":
        if not isinstance(urn, Urn):
            instance = self.parse(urn)
        else:
            instance = urn
        return instance


class NoResourceTypeError(Exception):
    def __str__(self):
        return "Unable to generate full urn because resource type is not provided. Use get_short_urn() method."
