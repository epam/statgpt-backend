from sdmx.message import StructureMessage

from common.auth.auth_context import AuthContext
from common.config import DialConfig
from common.data.quanthub.config import QuanthubSdmxDataSourceConfig
from common.data.quanthub.v21.qh_sdmx_client import AsyncQuanthubClient
from common.data.sdmx.common.config import SdmxConfig

# TODO: move to channel config?
HIERARCHIES = {
    "country_groups": dict(agency_id="IMF", resource_id="COUNTRY_GROUPS", version="1.2.0"),
}


SELECTED_HIERARCHIES = {
    'Advanced Economies': False,
    'ASEAN': False,
    'APEC': False,
    'Benelux': False,
    'BRIC': False,
    'BCEAO': False,
    'BEAC': False,
    'ECCU': False,
    'Euro Area': False,
    'European Union': False,
    'G7': False,
    'G20': False,
    'GCC': False,
    'MECA': False,
    'MENA': False,
    'SSA': False,
}


# perhaps use this class instead of dict[str, list[str]]
# class Hierarchy(BaseModel):
#     name: str
#     children: list[str]


class HierarchyAuthContext(AuthContext):
    """Auth context for hierarchy loading."""

    @property
    def is_system(self) -> bool:
        return True

    @property
    def dial_access_token(self) -> str | None:
        return None

    @property
    def api_key(self) -> str:
        return DialConfig.get_api_key().get_secret_value()


class HierarchiesLoader:
    _cache = {}
    _locale = 'en'

    @classmethod
    async def get_hierarchy(cls, hierarchy_name: str) -> dict[str, list[str]]:
        if hierarchy_name not in cls._cache:
            res = await cls._load_hierarchies_from_sdmx(hierarchy_name)
            cls._cache[hierarchy_name] = cls._filter_hierarchy(res)

        return cls._cache[hierarchy_name]

    @classmethod
    async def _load_hierarchies_from_sdmx(cls, hierarchy_name: str) -> dict[str, list[str]]:
        if hierarchy_name not in HIERARCHIES:
            raise ValueError(f"Unknown hierarchy: {hierarchy_name}")

        config = QuanthubSdmxDataSourceConfig(
            apiKey="$env:{QH_UAT_API_KEY}",
            sdmxConfig=SdmxConfig(
                id="QH_UAT",
                name="QuantHub UAT (Lite) SDMX Registry, Workspace: StatGPTDemo",
                url="https://apim-imfeid-dev-01.azure-api.net/statgpt-demo-sdmx",
            ),
        )
        auth_context = HierarchyAuthContext()
        client: AsyncQuanthubClient = AsyncQuanthubClient.from_config(config, auth_context)

        resp = await client.hierarchicalcodelist(
            **HIERARCHIES.get(hierarchy_name),
            params={"references": "children"},
            use_cache=True,
        )

        return await cls._parse_sdmx_response(resp)

    @classmethod
    async def _parse_sdmx_response(cls, response: StructureMessage) -> dict[str, list[str]]:
        res = {}

        for h in response.hierarchical_codelist[0].hierarchy[0].codes.values():
            if not h.child:
                continue

            name = h.code.name.localizations.get(cls._locale)
            res[name] = [c.code.name.localizations.get(cls._locale) for c in h.child]

        return res

    @staticmethod
    def _is_good_hierarchy(hierarchy: str) -> bool:
        for h, full_match in SELECTED_HIERARCHIES.items():
            if full_match and h == hierarchy:
                return True
            if not full_match and h in hierarchy:
                return True
        return False

    @classmethod
    def _filter_hierarchy(cls, items: dict[str, list[str]]) -> dict[str, list[str]]:
        return {k: v for k, v in items.items() if cls._is_good_hierarchy(k)}
