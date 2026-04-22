def generate_python_query_body(
    provider: str, flow_ref: str, key: str, params: dict, suffix: str = ""
) -> str:
    return f'''\
provider{suffix} = sdmx.Client("{provider}")
data_msg{suffix} = provider{suffix}.data(
    "{flow_ref}",
    key="{key}",
    params={params}
)'''
