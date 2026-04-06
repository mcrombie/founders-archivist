from dataclasses import dataclass, asdict, field
from typing import List


@dataclass
class HistoricalDocument:
    doc_id: str
    title: str
    author: str
    recipient: str
    date: str
    year: int
    collection: str
    document_type: str
    source_url: str
    text: str
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)