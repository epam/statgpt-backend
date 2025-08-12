from sdmx.model import common
from sdmx.reader.xml.v21 import Reader


def __ignore_formal_levels():
    @Reader.end("str:Hierarchy")
    def _h(reader: Reader, elem):
        cls = reader.class_for_tag(elem.tag)
        return reader.nameable(
            cls,
            elem,
            has_formal_levels=False,  # disable formal levels
            codes={c.id: c for c in reader.pop_all(common.HierarchicalCode)},
            level=reader.pop_single(common.Level),
        )


def __ignore_none_tag():
    # xml reader fails with NotImplemented when faces <None/> tag
    Reader.parser["None", "start"] = None
    Reader.parser["None", "end"] = None


def __ignore_group_tag():
    @Reader.end(":Group")
    def _group_2(reader, elem):
        pass


def __apply_sdmx_extensions():
    __ignore_none_tag()
    __ignore_formal_levels()
    __ignore_group_tag()


def init_qh_sdmx_extensions():
    __apply_sdmx_extensions()
