"""
Input: AI document section target (str)

For each folder in data/0.raw containing ai_document and combined_chunks, create a new JSON in a folder:
data/1.pre-processed/section_{section_target}/{tender_hash}.json

The file created should:
- Take the AI document target section and use regex to extract the chunk ids [chunk_[...]]
CHUNK_REF_PATTERN = re.compile(r"\[(chunk_.*?)\]")
- Create a copy of the combined chunks for that tender and append at the end of that JSON a list of positives (the chunks found in the target section)
- Place this new file in the data/1.pre-processed/section_{section_target}/{tender_hash}.json

"""

import re
import json
from pathlib import Path

CHUNK_REF_PATTERN = re.compile(r"\[(chunk_.*?)\]")


class TenderPreprocessor:
    def __init__(self, section_target: str):
        self.section_target = section_target
        self.raw_data_path = Path("data/0.raw")
        section_slug = section_target.split(".")[0]
        self.output_path = Path(f"data/1.pre-processed/section_{section_slug}")
        self.output_path.mkdir(parents=True, exist_ok=True)

    def process_tender(self, tender_path: Path):
        tender_hash = tender_path.name
        ai_document_path = tender_path / "ai_document.json"
        combined_chunks_path = tender_path / "combined_chunks.json"

        if not ai_document_path.exists() or not combined_chunks_path.exists():
            print(f"Skipping {tender_hash}: missing required files")
            return

        try:
            with open(ai_document_path, "r", encoding="utf-8") as f:
                ai_document = json.load(f)

            with open(combined_chunks_path, "r", encoding="utf-8") as f:
                combined_chunks = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Skipping {tender_hash}: JSON decode error - {e}")
            return

        if self.section_target not in ai_document:
            print(f"Skipping {tender_hash}: section '{self.section_target}' not found")
            return

        section_content = ai_document[self.section_target]
        chunk_ids = CHUNK_REF_PATTERN.findall(section_content)

        output_data = {
            "chunks": combined_chunks,
            "positives": list(set(chunk_ids))
        }

        output_file = self.output_path / f"{tender_hash}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Processed {tender_hash}: {len(chunk_ids)} positive chunks")

    def process_all_tenders(self):
        tender_folders = [f for f in self.raw_data_path.iterdir() if f.is_dir()]
        print(f"Found {len(tender_folders)} tender folders")

        for tender_folder in tender_folders:
            self.process_tender(tender_folder)


if __name__ == "__main__":
    import sys
    print(len(sys.argv))

    section_target = sys.argv[1] if len(sys.argv) >1 else "5. CRITERIOS DE ADJUDICACIÓN"
    preprocessor = TenderPreprocessor(section_target)
    preprocessor.process_all_tenders()
