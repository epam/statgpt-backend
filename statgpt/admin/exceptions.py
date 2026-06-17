class AdminServiceError(Exception):
    """Base class for admin service errors mapped to HTTP responses in routers."""


class DatasetInUseError(AdminServiceError):
    def __init__(self, *, dataset_id: int, dataset_title: str, channel_count: int) -> None:
        self.dataset_id = dataset_id
        self.dataset_title = dataset_title
        self.channel_count = channel_count
        super().__init__(
            f"Dataset '{dataset_title}' (id={dataset_id}) is used in {channel_count} channel(s)"
        )
