from dataclasses import dataclass


class AdminServiceError(Exception):
    """Base class for admin service errors mapped to HTTP responses in routers."""


@dataclass(frozen=True)
class BlockingDataset:
    dataset_id: int
    dataset_title: str
    channel_count: int

    @property
    def channels_label(self) -> str:
        suffix = "" if self.channel_count == 1 else "s"
        return f"{self.channel_count} channel{suffix}"

    @property
    def usage(self) -> str:
        return f"'{self.dataset_title}' ({self.channels_label})"


class DatasetInUseError(AdminServiceError):
    def __init__(self, blocking_datasets: list[BlockingDataset]) -> None:
        self.blocking_datasets = blocking_datasets
        super().__init__(f"Dataset(s) still used in channels: {self.usage_summary}")

    @property
    def usage_summary(self) -> str:
        return ", ".join(ds.usage for ds in self.blocking_datasets)
