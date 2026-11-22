import os
from pathlib import Path


class _Names:
    def words(self, fileid):
        d = os.environ.get("NLTK_NAMES_DIR")
        if d and (Path(d) / fileid).exists():
            return [w.strip() for w in (Path(d) / fileid).read_text().splitlines() if w.strip()]
        return []


names = _Names()
