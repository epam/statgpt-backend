import uuid
from typing import Any

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Mapped, mapped_column, relationship

from statgpt.common.schemas import JobType, PreprocessingStatusEnum
from statgpt.common.settings.elastic import ElasticSearchSettings
from statgpt.common.settings.langchain import langchain_settings
from statgpt.common.utils import DateMixin, IdMixin

from .database import Base

_elasticsearch_settings = ElasticSearchSettings()


class DefaultBase(IdMixin, DateMixin, Base):
    __abstract__ = True


class Channel(DefaultBase):
    __tablename__ = "channels"

    title: Mapped[str]
    description: Mapped[str]
    deployment_id: Mapped[str] = mapped_column(unique=True)
    llm_model: Mapped[str] = mapped_column(
        default=langchain_settings.embedding_default_model.value
    )
    details: Mapped[dict[str, Any]] = mapped_column(type_=postgresql.JSONB)

    # ~~~~~ Relationships ~~~~~

    mapped_datasets: Mapped[list["ChannelDataset"]] = relationship(
        back_populates="channel", cascade="all, delete-orphan"
    )
    glossary_terms: Mapped[list["GlossaryTerm"]] = relationship(
        "GlossaryTerm", back_populates="channel", cascade="all, delete-orphan"
    )
    # NOTE: Jobs can exist without a channel, so the option `delete-orphan` is not used here.
    jobs: Mapped[list["Job"]] = relationship(
        "Job", back_populates="channel", cascade="all"
    )

    # ~~~~~ Properties ~~~~~

    def __repr__(self) -> str:
        return f"Channel(id={self.id!r}, title={self.title!r})"

    @property
    def indicator_table_name(self) -> str:
        return f"Indicators_{self.id}"

    @property
    def available_dimensions_table_name(self) -> str:
        return f"AvailableDimensions_{self.id}"

    @property
    def special_dimensions_table_name(self) -> str:
        return f"SpecialDimensions_{self.id}"

    @property
    def matching_index_name(self) -> str:
        return f"{_elasticsearch_settings.matching_index}_{self.id}"

    @property
    def indicators_index_name(self) -> str:
        return f"{_elasticsearch_settings.indicators_index}_{self.id}"


class DataSourceType(DefaultBase):
    __tablename__ = "data_source_types"

    name: Mapped[str]
    description: Mapped[str] = mapped_column(default="")

    # relationships
    data_sources: Mapped[list["DataSource"]] = relationship(
        back_populates="type", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"DataSourceType(id={self.id!r}, name={self.name!r})"


class DataSource(DefaultBase):
    __tablename__ = "data_sources"

    title: Mapped[str]
    description: Mapped[str]
    type_id: Mapped[int] = mapped_column(ForeignKey("data_source_types.id"))
    details: Mapped[dict[str, Any]] = mapped_column(type_=postgresql.JSONB)

    # relationships
    type: Mapped[DataSourceType] = relationship(back_populates="data_sources")
    datasets: Mapped[list["DataSet"]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"DataSource(id={self.id!r}, type_id={self.type_id!r}, title={self.title!r})"


class DataSet(DefaultBase):
    __tablename__ = "datasets"

    # TODO: use id_ as primary_key
    id_: Mapped[uuid.UUID] = mapped_column(
        type_=postgresql.UUID(as_uuid=True), unique=True
    )
    source_id: Mapped[int] = mapped_column(ForeignKey("data_sources.id"))
    title: Mapped[str]
    details: Mapped[dict[str, Any]] = mapped_column(type_=postgresql.JSONB)

    # relationships
    source: Mapped[DataSource] = relationship(back_populates="datasets")
    mapped_channels: Mapped[list["ChannelDataset"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"DataSet(id_={self.id_!r}, source_id={self.source_id!r}, title={self.title!r})"


class ChannelDataset(DefaultBase):
    __tablename__ = "channel_datasets"

    channel_id: Mapped[int] = mapped_column(ForeignKey("channels.id"))
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))

    clearing_status: Mapped[PreprocessingStatusEnum] = mapped_column(
        default=PreprocessingStatusEnum.NOT_STARTED
    )
    """The status of the data clearing task run after reindexing is complete."""

    # relationships
    channel: Mapped[Channel] = relationship(back_populates="mapped_datasets")
    dataset: Mapped[DataSet] = relationship(back_populates="mapped_channels")
    versions: Mapped[list["ChannelDatasetVersion"]] = relationship(
        back_populates="channel_dataset", cascade="all, delete"
    )

    def __repr__(self) -> str:
        return f"ChannelDataset(id={self.id!r}, channel_id={self.channel_id!r}, dataset_id={self.dataset_id!r})"


class ChannelDatasetVersion(DefaultBase):
    __tablename__ = "channel_dataset_versions"
    __table_args__ = (
        UniqueConstraint(
            "channel_dataset_id",
            "version",
            name="uix_unique_version_for_channel_dataset",
        ),
    )

    channel_dataset_id: Mapped[int] = mapped_column(ForeignKey("channel_datasets.id"))
    version: Mapped[int] = mapped_column(
        default=0
    )  # will be auto-incremented by trigger
    preprocessing_status: Mapped[PreprocessingStatusEnum]
    pointer_to: Mapped[int | None] = mapped_column(
        ForeignKey("channel_dataset_versions.id", ondelete="SET NULL"), default=None
    )

    creation_reason: Mapped[str]
    reason_for_failure: Mapped[str | None] = mapped_column(default=None)

    structure_metadata: Mapped[dict | None] = mapped_column(
        type_=postgresql.JSONB, default=None
    )
    structure_hash: Mapped[str | None] = mapped_column(type_=String(10), default=None)
    # Data hashes:
    indicator_dimensions_hash: Mapped[str | None] = mapped_column(
        type_=String(10), default=None
    )
    non_indicator_dimensions_hash: Mapped[str | None] = mapped_column(
        type_=String(10), default=None
    )
    special_dimensions_hash: Mapped[str | None] = mapped_column(
        type_=String(10), default=None
    )
    # Config hash:
    indexing_config_hash: Mapped[str | None] = mapped_column(
        type_=String(10), default=None
    )
    # Resolved configuration at indexing time (e.g. dynamic URN version resolved to concrete version)
    resolved_config: Mapped[dict | None] = mapped_column(
        type_=postgresql.JSONB, default=None
    )

    # relationships
    channel_dataset: Mapped[ChannelDataset] = relationship(back_populates="versions")
    pointer = relationship(
        "ChannelDatasetVersion",
        remote_side="ChannelDatasetVersion.id",
        back_populates="pointing_versions",
        cascade="all",
        passive_deletes=True,
    )
    pointing_versions = relationship(
        "ChannelDatasetVersion",
        remote_side="ChannelDatasetVersion.pointer_to",
        back_populates="pointer",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return (
            f"ChannelDatasetVersion(id={self.id!r}, channel_dataset_id={self.channel_dataset_id!r},"
            f" version={self.version!r}, preprocessing_status={self.preprocessing_status!r},"
            f" pointer_to={self.pointer_to!r})"
        )


class Job(DefaultBase):
    """Import/export job."""

    __tablename__ = "jobs"

    type: Mapped[JobType]
    status: Mapped[PreprocessingStatusEnum]
    file: Mapped[str | None] = mapped_column(default=None)  # path or url
    channel_id: Mapped[int | None] = mapped_column(ForeignKey("channels.id"))
    reason_for_failure: Mapped[str | None] = mapped_column(default=None)

    # relationships
    channel: Mapped[Channel | None] = relationship(back_populates="jobs")


class GlossaryTerm(DefaultBase):
    """A model representing a single term from the glossary.

    For details, see the following documentation page:
    https://gitlab.deltixhub.com/Deltix/migapp/talk-to-your-data/documentation/-/blob/main/designs/glossary_of_terms.md?ref_type=heads#data
    """

    __tablename__ = "glossary_terms"

    channel_id: Mapped[int] = mapped_column(ForeignKey("channels.id"))

    term: Mapped[str]
    definition: Mapped[str]

    #  ~~~ Metadata for the term: ~~~
    # It might be better to create a JSONB `details` field and move these fields there.
    domain: Mapped[str]
    source: Mapped[str]

    # relationships
    channel: Mapped[Channel] = relationship(back_populates="glossary_terms")

    def __repr__(self) -> str:
        return f"GlossaryTerm(id={self.id!r}, channel_id={self.channel_id!r}, term={self.term!r})"
