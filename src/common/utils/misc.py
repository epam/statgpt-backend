import base64
import hashlib
import itertools
import subprocess
import typing as t
import uuid


def batched(iterable: t.Iterable, n: int):
    """Batch data from the iterable into tuples of length n. The last batch may be shorter than n.

    In Python 3.12 and later, use the built-in `itertools.batched` function.

    Example:
        batched('ABCDEFG', 3) → ['A', 'B', 'C'], ['D', 'E', 'F'], ['G']
    """

    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, n)):
        yield batch


def get_last_commit_hash_for(path: str) -> str:
    """Get latest git commit hash for a specified directory/file path, not for the entire repository."""
    proc = subprocess.run(
        ["git", "log", "-n", "1", "--pretty=format:%H", path], capture_output=True, text=True
    )
    return commit_hash if (commit_hash := proc.stdout) is not None else ""


def str2bool(var: str) -> bool:
    return var.strip().lower() == "true"


def secret_2_safe_str(secret: str | None) -> str | None:
    """To securely log secrets"""

    if secret is None:
        return secret

    if len(secret) < 5:
        return "*" * len(secret)

    if len(secret) < 7:
        return secret[:1] + "***" + secret[-1:]

    if len(secret) < 9:
        return secret[:2] + "***" + secret[-2:]

    return secret[:3] + "***" + secret[-3:]


def create_base64_uuid():
    """uuid string is too long for some uses (filename). shorten by encoding to base64"""
    uuid_ = uuid.uuid4()
    # use urlsafe_b64encode to avoid using + and / chars
    uuid_b64_str = base64.urlsafe_b64encode(uuid_.bytes).decode()
    # uuid string holds 128 bits and always ends with '==':
    #   ceil(128 / 6) = 22, and base64 strings must be a multiple of 4 characters,
    #   thus, add 2 '=' padding symbols
    # remove padding symbols
    res = uuid_b64_str.rstrip('=')
    return res


def get_file_hash(fp: str, hashfunc_factory=hashlib.md5, chunk_size=1024 * 1024):
    """'
    Compute hash of a file. File is read in chunks of size 'chunk_size'
    """
    hashfunc = hashfunc_factory()
    with open(fp, "rb") as fin:
        while True:
            chunks = fin.read(chunk_size)
            if chunks:
                hashfunc.update(chunks)
            else:
                break
    return hashfunc.hexdigest()


def argparse_parse_int_or_none(val: str) -> t.Optional[int]:
    if not val:
        return None
    return int(val)


def string_split_snowball(s: str, sep: str) -> t.Generator[str, None, None]:
    """
    'a/b/c', '/' -> ['a', 'a/b', 'a/b/c']
    'abc', '/' -> ['abc']
    """
    search_start_ix = -1
    while True:
        found_ix = s.find("/", search_start_ix + 1)
        if found_ix < 0:
            yield s
            break
        substr = s[:found_ix]
        yield substr
        search_start_ix = found_ix
