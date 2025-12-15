import asyncio
import base64
import hashlib
import itertools
import subprocess
import uuid
import zlib
from collections.abc import Generator, Iterable


def batched(iterable: Iterable, n: int):
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


def crc32_hash(data: str) -> int:
    """Compute CRC32 hash of a string and return it as a positive integer."""
    return zlib.crc32(data.encode("utf-8")) & 0xFFFFFFFF


def crc32_hash_incremental(values: Iterable[str]) -> int:
    """
    Compute CRC32 hash incrementally from an iterable of strings.

    This avoids creating a large intermediate string, reducing memory usage
    and making the operation more efficient for large lists.

    Args:
        values: iterable of strings to hash (in sorted order)

    Returns:
        CRC32 hash as a positive integer
    """
    crc = 0
    for value in values:
        # Hash each value with newline separator
        crc = zlib.crc32(f"{value}\n".encode("utf-8"), crc)
    return crc & 0xFFFFFFFF


async def crc32_hash_incremental_async(values: list[str]) -> int:
    """
    Async version of crc32_hash_incremental.

    Offloads the blocking hash computation to a thread pool to avoid
    blocking the asyncio event loop during large dataset processing.

    Args:
        values: Sorted list of strings to hash

    Returns:
        CRC32 hash as a positive integer
    """
    return await asyncio.to_thread(crc32_hash_incremental, values)


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


def argparse_parse_int_or_none(val: str) -> int | None:
    if not val:
        return None
    return int(val)


def string_split_snowball(s: str, sep: str) -> Generator[str, None, None]:
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
