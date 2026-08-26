# TODO: Find a better way to store these parameters:
class DataQueryParameters:
    RESPONSE_FIELD = "data_query_response"
    STATE = "data_query_state"
    MCP_PAYLOAD = "data_query_mcp_payload"
    EVAL_ATTACHMENT = "data_query_eval_attachment"
    DATASET_CHOICES = "dataset_choices"
    # Assigned by the search-preparation chain, and read again by the discovery fallback so it
    # does not have to extract the request's countries a second time.
    COUNTRY_NAMED_ENTITIES = "country_named_entities"
