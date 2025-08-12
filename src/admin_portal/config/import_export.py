class JobsConfig:
    """Import/Export jobs configuration."""

    JOBS_RETENTION_HOURS = 24
    """The default hours after which jobs are cleared when `clear_jobs` is called."""

    CSV_COLUMNS = ['page_content', 'metadata', 'embeddings']
    ENCODING = 'utf-8'

    # Dial folders:
    DIAL_EXPORT_FOLDER = 'exports'
    DIAL_IMPORT_FOLDER = 'imports'

    # File and folder names to store exported data:
    CHANNEL_FILE = 'channel.yaml'
    DATASETS_FILE = 'datasets.yaml'
    DATA_SOURCES_FILE = 'data_sources.yaml'
    GLOSSARY_TERMS_FILE = 'glossary_terms.csv'
    DIAL_FILES_FOLDER = 'dial_files'

    # Elasticsearch index folders:
    ES_MATCHING_DIR = 'es_matching'
    ES_INDICATORS_DIR = 'es_indicators'

    # Exported fields:
    CHANNEL_FIELDS = {"deployment_id", "title", "description", "llm_model", "details"}
    DATASET_FIELDS = {"id_", "title", "details"}  # and dynamic 'dataSource' field
    DATA_SOURCE_FIELDS = {"title", "description", "type_id", "details"}
