"""
Fetches data and places them in raw content folders at data/0.raw

1. Iterates rows from target CSV (path as input)
2. For each row (tender) - retrieve the tender_hash
3. Check if path exists at  data/0.raw/{tender_hash} and folder contains /ai_document.json and /combined_chunks.json
4. If exists continue to next
5. Else - take ai_document json from CSV and create a new json file /ai_document.json with contents.
6. Fetch from aws S3 the combined chunks using the aws client.

Iterate over all rows and the csv to complete task.
Example csv at @src/data/public_tender_info_export_2025-09-30_212154.csv

"""
import sys
import csv
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
load_dotenv(project_root / ".env")

from src.utils.aws_client import AWSS3Client
from src.data.schemas import ChunkType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self, csv_path: str, output_dir: str = "data/0.raw"):
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.aws_client = AWSS3Client()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _check_tender_exists(self, tender_hash: str) -> bool:
        """Check if tender folder exists with required files."""
        tender_path = self.output_dir / tender_hash
        ai_doc_path = tender_path / "ai_document.json"
        chunks_path = tender_path / "combined_chunks.json"

        return tender_path.exists() and ai_doc_path.exists() and chunks_path.exists()

    def _save_ai_document(self, tender_hash: str, ai_document: str) -> None:
        """Save ai_document from CSV to JSON file."""
        tender_path = self.output_dir / tender_hash
        tender_path.mkdir(parents=True, exist_ok=True)

        ai_doc_path = tender_path / "ai_document.json"

        try:
            ai_doc_data = json.loads(ai_document)
        except json.JSONDecodeError:
            ai_doc_data = {"content": ai_document}

        with open(ai_doc_path, 'w', encoding='utf-8') as f:
            json.dump(ai_doc_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved ai_document for {tender_hash}")

    def _fetch_and_save_chunks(self, tender_hash: str) -> bool:
        """Fetch combined chunks from S3 and save to JSON file. Returns True if successful."""
        tender_path = self.output_dir / tender_hash
        tender_path.mkdir(parents=True, exist_ok=True)

        chunks_path = tender_path / "combined_chunks.json"

        try:
            chunks = self.aws_client.get_tender_chunks(
                tender_hash=tender_hash,
                chunk_type=ChunkType.ALL
            )

            if chunks:
                with open(chunks_path, 'w', encoding='utf-8') as f:
                    json.dump(chunks, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved {len(chunks)} chunks for {tender_hash}")
                return True
            else:
                logger.warning(f"No chunks found for {tender_hash}")
                return False
        except Exception as e:
            logger.error(f"Error fetching chunks for {tender_hash}: {e}")
            return False

    def process_csv(self) -> None:
        """Process CSV file and fetch data for each tender."""
        with open(self.csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                tender_hash = row.get('tender_hash')
                ai_document = row.get('ai_document', '')

                if not tender_hash:
                    logger.warning("Row missing tender_hash, skipping")
                    continue

                if self._check_tender_exists(tender_hash):
                    logger.info(f"Tender {tender_hash} already exists, skipping")
                    continue

                logger.info(f"Processing tender {tender_hash}")

                if ai_document:

                    chunks_success = self._fetch_and_save_chunks(tender_hash)

                    if chunks_success:
                        self._save_ai_document(tender_hash, ai_document)


def main():
    csv_path = "src/data/public_tender_info_export_2025-09-30_212154.csv"
    fetcher = DataFetcher(csv_path)
    fetcher.process_csv()


if __name__ == "__main__":
    main()
