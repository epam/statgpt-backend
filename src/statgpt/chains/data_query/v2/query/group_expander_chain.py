from aidial_sdk.chat_completion import Stage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field

from common.config import multiline_logger as logger
from common.schemas import LLMModelConfig
from common.settings.dial import dial_settings
from common.utils.models import get_chat_model
from statgpt.chains.parameters import ChainParameters
from statgpt.utils.callbacks import StageCallback

FOUND_GROUP_DESCRIPTION = """\
A dictionary in which the key is the group ID from the list of available groups,\
 and the value is one of the following strings:
* "whole" if the user mentioned this group of countries as a whole\
 (e.g., "<group> as a group", "<group> region", etc.);
\
* "members" - the user wants to receive information for each participant, and not for the entire group\
 (e.g., "<group> members", "<group> countries", "countries of <group>", "members of <group>",
  "nations in <group>", "compare <something> for <group>", etc.);
\
* "both" - user is interested in both the group of countries as a whole and its participants.
\
If user did not explicitly refer to members of a country group (like "X members" or "Y countries"),\
 assume it referred group as a whole. If there are no matching groups, you must provide an empty dictionary.\
"""

TEMPLATE_DESCRIPTION = """\
A user request in which found country groups have been replaced with placeholders.\
 Don't change anything else in the user's request.\
 If no country groups are found, repeat the user request without changes.\

The placeholder must have the following format: `<GROUP_{{ID}}>`\
 where `{{ID}}` is the ID of the group.\
"""

ABSENT_GROUP_DESCRIPTION = """\
List of country group names mentioned in the "template" field.\
 This means that these groups are not in the list of available groups and have not been replaced by placeholders.\
 If there are no such groups, provide an empty list.\
"""


class GroupExpanderLLMResponse(BaseModel):
    found_groups: dict[int, str] = Field(description=FOUND_GROUP_DESCRIPTION)
    template: str = Field(description=TEMPLATE_DESCRIPTION)
    absent_groups: list[str] = Field(description=ABSENT_GROUP_DESCRIPTION)


class GroupExpanderChain:
    def __init__(
        self,
        llm_model_config: LLMModelConfig,
        system_prompt: str,
        fallback_prompt: str,
        llm_api_base: str | None = None,
    ):
        self._system_prompt = system_prompt
        self._fallback_prompt = fallback_prompt
        self._llm_api_base = llm_api_base or dial_settings.url
        self._llm_model_config = llm_model_config

    async def _get_country_groups(
        self, hierarchies: dict[str, list[str]] | None = None
    ) -> list[dict]:
        return [{"ID": i, "Name": k} for i, k in enumerate(hierarchies.keys(), start=1)]

    async def _parse_response(self, inputs: dict) -> dict:
        response: GroupExpanderLLMResponse = inputs['expanded_groups_response']
        hierarchies: dict[str, list[str]] = inputs['hierarchies']
        country_groups: list[dict] = inputs['country_groups']
        auth_context = inputs['auth_context']

        logger.info(f"{response!r}")

        if not response.found_groups and not response.absent_groups:
            inputs['query_with_expanded_groups'] = ''
            inputs['expanded_groups'] = {}
            return inputs

        res = response.template
        expanded_groups = {}
        for group_id, status in response.found_groups.items():
            group_name = next((g['Name'] for g in country_groups if g['ID'] == group_id), None)
            if group_name is None:
                msg = f"Hallucination! LLM selected non-existing group: {group_id}. {response=}"
                logger.error(msg)
                # TODO: can think of graceful handling here, instead of raising an error
                raise RuntimeError(msg)

            placeholder = f"<GROUP_{group_id}>"

            if status == 'whole':
                expanded_groups[group_name] = 'left unchanged'
                res = res.replace(placeholder, group_name)
                continue
            expanded_groups[group_name] = 'expanded with reference hierarchy'

            countries = ', '.join(hierarchies[group_name])
            if status == 'members':
                res = res.replace(placeholder, countries)
            elif status == 'both':
                res = res.replace(placeholder, f"{group_name}, {countries}")
            else:
                msg = (
                    f"Hallucination! LLM selected invalid group expansion '{status=}'. {response=}"
                )
                logger.error(msg)
                raise RuntimeError(msg)

        if response.absent_groups:
            expanded_groups.update((g, 'processed by LLM') for g in response.absent_groups)
            res = await self.run_fallback(auth_context, res, response.absent_groups)

        inputs['query_with_expanded_groups'] = res
        inputs['expanded_groups'] = expanded_groups
        return inputs

    async def run_fallback(self, auth_context, res_query: str, absent_groups: list[str]) -> str:
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self._fallback_prompt),
                ("user", res_query),
            ],
        )
        fallback_model = get_chat_model(
            api_key=auth_context.api_key,
            azure_endpoint=self._llm_api_base,
            model_config=self._llm_model_config,
        )
        logger.info(
            f"{self.__class__.__name__} (fallback chain) using LLM model: "
            f"{self._llm_model_config.deployment.deployment_id}"
        )

        chain = prompt_template | fallback_model | StrOutputParser()

        return await chain.ainvoke(dict(absent_groups=absent_groups))

    @staticmethod
    async def _populate_group_expander(stage: Stage, inputs: dict):
        query = inputs.get("query_with_expanded_groups", "")
        expanded_groups = inputs.get("expanded_groups", {})

        if query:
            stage.append_content(f"Query with unpacked groups: `{query}`  \n")

        if expanded_groups:
            stage.append_content("  \nFound groups:\n")
            for group, comment in expanded_groups.items():
                stage.append_content(f"* {group} ({comment})\n")

    async def create_chain(self, inputs: dict) -> Runnable:
        raise NotImplementedError("GroupExpanderChain is outdated and needs to be reimplemented")

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self._system_prompt),
                ("user", ChainParameters.get_query(inputs)),
            ],
        )

        auth_context = ChainParameters.get_auth_context(inputs)
        parser = PydanticOutputParser(pydantic_object=GroupExpanderLLMResponse)

        model = get_chat_model(
            api_key=auth_context.api_key,
            azure_endpoint=self._llm_api_base,
            model_config=self._llm_model_config,
        )
        logger.info(
            f"{self.__class__.__name__} (main chain) using LLM model: "
            f"{self._llm_model_config.deployment.deployment_id}"
        )

        # hierarchies = await HierarchiesLoader.get_hierarchy("country_groups")
        hierarchies = {}
        country_groups = await self._get_country_groups(hierarchies)

        chain = (
            RunnablePassthrough.assign(
                country_groups=lambda _: country_groups,
                format_instructions=lambda _: parser.get_format_instructions(),
            )
            | prompt_template
            | model.with_structured_output(GroupExpanderLLMResponse, method="json_mode")
            | RunnableLambda(lambda res: dict(expanded_groups_response=res))
            | RunnablePassthrough.assign(
                auth_context=lambda _: auth_context,
                hierarchies=lambda _: hierarchies,
                country_groups=lambda _: country_groups,
            )
            | self._parse_response
        )
        return chain.with_config(
            config=RunnableConfig(
                callbacks=[
                    StageCallback(
                        "Unpacking Groups", self._populate_group_expander, debug_only=True
                    ),
                ]
            )
        ) | RunnableLambda(lambda inputs: inputs['query_with_expanded_groups'])
