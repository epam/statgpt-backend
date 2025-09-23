import base64
import csv
import json
import os
import re
from enum import StrEnum
from pathlib import Path

import yaml

# custom yaml dumper configuration


def _yaml_multiline_str_representer(dumper, data):
    """
    Emit strings with a newline as a literal block scalar ('|'),
    otherwise use the default scalar style.

    NOTE: will not work if string ends with trailing whitespaces.
    as a workaround, you can strip such strings before dumping.
    """
    tag = 'tag:yaml.org,2002:str'
    if '\n' in data:
        style = '|'  # PyYAML adds the “-” automatically when needed
        return dumper.represent_scalar(tag, data, style=style)
    return dumper.represent_scalar(tag, data)


# NOTE: alternatively, we can use custom dumper subclassed from yaml.SafeDumper:
#
# class CustomYamlSafeDumper(yaml.SafeDumper):
# pass
#
# but in this case, yaml.safe_dump will not work - we will need to use yaml.dump instead.
# so I decided with updating default yaml.SafeDumper inplace instead.

yaml.SafeDumper.add_representer(str, _yaml_multiline_str_representer)

# treat every StrEnum object like a regular string when dumping to YAML
yaml.SafeDumper.add_multi_representer(
    StrEnum,
    yaml.representer.SafeRepresenter.represent_str,
)


INVALID_PATH_CHARS_PATTERN = r'[<>:"/\\|?*\x00-\x1F]'


def is_file_or_folder_name_valid(name: str, length_limit: int = 100) -> bool:
    """Check if a file or folder name is valid (no invalid characters, length restrictions).

    Microsoft Windows has a MAX_PATH limit of ~256 characters.
    If the length of the path and filename combined exceed ~256 characters there will be issues.
    Here we limit the length of the name of one file or folder to 100 characters,
    but keep in mind that the Windows limit is for the entire path.
    """
    if len(name) > length_limit:
        return False
    if re.search(INVALID_PATH_CHARS_PATTERN, name):
        return False
    return True


def escape_invalid_filename_chars(filename: str) -> str:
    escaped_filename = re.sub(INVALID_PATH_CHARS_PATTERN, '_', filename)
    return escaped_filename


def change_file_extension(fp: str, new_extension: str, apply: bool = False) -> str:
    """Change the extension of a file. If `apply` is True, the file on disk will be renamed.

    Returns:
         the new file path with the new extension.
    """
    if not new_extension.startswith("."):
        new_extension = "." + new_extension
    new_fp = os.path.splitext(fp)[0] + new_extension

    if apply:
        os.rename(fp, new_fp)

    return new_fp


def read_txt(fp: str) -> str:
    with open(fp) as fin:
        text = fin.read()
    return text


def write_text(text, fp, mode="w", encoding="utf-8"):
    with open(fp, mode=mode, encoding=encoding) as fout:
        fout.write(text)


def read_csv_as_dict_list(fp: str, encoding="utf-8") -> list[dict[str, str]]:
    with open(fp, encoding=encoding, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader]


def write_csv_from_dict_list(
    data: list[dict[str, str]], fp: str, mode="w", encoding="utf-8"
) -> None:
    if not data:
        raise ValueError("No data to write")

    with open(fp, mode=mode, newline='', encoding=encoding) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())

        writer.writeheader()
        writer.writerows(data)


def read_yaml(fp: Path | str, encoding="utf-8"):
    with open(fp, encoding=encoding) as fin:
        data = yaml.safe_load(fin)
    return data


def write_yaml_to_stream(data, stream=None, indent=2, width=10000, **kwargs):
    return yaml.safe_dump(
        data,
        stream,
        indent=indent,
        width=width,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        **kwargs,
    )


def write_yaml(data, fp: Path | str, mode="w", encoding="utf-8", indent=2, width=10000, **kwargs):
    with open(fp, mode=mode, encoding=encoding) as fout:
        _ = write_yaml_to_stream(data=data, stream=fout, indent=indent, width=width, **kwargs)


def write_bytes(data, fp, mode="wb"):
    dp = os.path.dirname(fp)
    if dp:
        os.makedirs(dp, exist_ok=True)
    with open(fp, mode=mode) as fout:
        fout.write(data)


def read_bytes(fp: str) -> bytes:
    with open(fp, "rb") as fin:
        data = fin.read()
    return data


def read_json(fp: str):
    with open(fp, encoding="utf-8") as fin:
        data = json.load(fin)
    return data


def read_json_lines(fp: str, encoding=None) -> list:
    with open(fp, encoding=encoding) as fin:
        lines = fin.readlines()
    items = list(map(json.loads, lines))
    return items


def write_json(
    obj,
    fp,
    mode="w",
    encoding='utf-8',
    ensure_ascii=False,
    indent: int | None = 2,
    add_newline=False,
):
    """
    'indent' controls the json indent.
    'add_newline' is useful when iteratively dumping json lines into a single file, e.g. in pipelines.

    to write objects in json-lines files (.jsonl) you need to set:
    `mode='a', indent=None, add_newline=True`.
    """
    with open(fp, mode=mode, encoding=encoding) as fout:
        if not add_newline:
            json.dump(obj, fout, ensure_ascii=ensure_ascii, indent=indent)
        else:
            s = json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent)
            fout.write(s + "\n")


def decode_file_from_base64_str(
    base64_str: str,
    output_path: str,
    encoding: str = "utf-8",
) -> None:
    """Decode a base64 string and write it to a file."""
    file_bytes = base64.b64decode(base64_str)
    write_bytes(file_bytes, output_path, mode="wb")
