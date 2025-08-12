import csv
from io import StringIO

import pandas as pd


def df_2_table_str(df: pd.DataFrame, delimiter='|') -> str:
    out = StringIO()

    fieldnames = df.columns
    writer = csv.DictWriter(
        out,
        fieldnames=fieldnames,
        delimiter=delimiter,
        # instead of quoting we opt for escaping special chars
        quoting=csv.QUOTE_NONE,
        escapechar='\\',
        # since we don't use quotes,
        # we need to override default quotechar
        # to avoid escaping it with escapechar.
        quotechar=None,
        # default is '\r\n'
        lineterminator='\n',
    )

    writer.writeheader()

    # write delimiter between header and body
    # out.write('|'.join('-' * df.shape[1]) + '\n')
    out.write('---\n')  # use a simple one

    recs = df.to_dict(orient='records')
    for item in recs:
        writer.writerow(item)  # type: ignore

    res = out.getvalue()
    out.close()
    return res


def pull_columns_to_front(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Reorder dataframe columns by pulling specified columns to the front"""
    # NOTE: do not use sets everywhere to preserve order within each group of columns
    df_cols_set = set(df.columns)
    first_cols = [col for col in columns if col in df_cols_set]
    first_cols_set = set(first_cols)
    rest_cols = [col for col in df.columns if col not in first_cols_set]
    return df[first_cols + rest_cols]


class BatchedDataFrame:
    """Access DataFrame in batches of fixed size."""

    _df: pd.DataFrame
    _n: int
    _batch_size: int

    def __init__(self, df: pd.DataFrame, batch_size: int):
        self._df = df
        self._batch_size = batch_size

    def __len__(self):
        return (self._df.shape[0] - 1) // self.batch_size + 1

    @property
    def batch_size(self):
        return self._batch_size

    def __getitem__(self, ix: int | tuple | list):
        if not isinstance(ix, (int, tuple, list)):
            raise TypeError(f'expected index to be int, tuple or list. got {type(ix)}: {ix}')
        if isinstance(ix, (tuple, list)):
            df_batches = [self[batch_ix] for batch_ix in ix]
            df_concat = pd.concat(df_batches, axis=0)
            return df_concat
        if ix >= len(self):
            raise IndexError(f'index {ix} is out of range')
        if ix < 0:
            raise ValueError(f'negative indices are not supported. got: {ix}')
        return self._df.iloc[self.batch_size * ix : self.batch_size * (ix + 1)]

    def __iter__(self):
        for ix in range(len(self)):
            yield self[ix]
