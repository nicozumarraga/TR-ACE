"""
Retrieves raw data from S3 and database in order to have raw training data.
"""
import os
import logging
import json
from datetime import datetime, timedelta
from typing import Optional, Union, List, Dict, Any
from enum import Enum
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

from src.data.schemas import ChunkType

# Load environment variables
load_dotenv()


class S3BucketType(Enum):
    """Enum for available S3 buckets."""
    LEGACY = "gober-tender-documents"  # Old bucket for backwards compatibility
    GLOBAL = "gober-app-global-bucket"  # New global bucket


class AWSS3Client:
    """Client for AWS S3 operations."""

    def __init__(self, bucket_type: S3BucketType = S3BucketType.GLOBAL):
        """Initialize AWS S3 client.

        Args:
            bucket_type: The S3 bucket to use (LEGACY or GLOBAL). Defaults to GLOBAL.
        """
        self.bucket_type = bucket_type
        self.bucket_name = bucket_type.value
        self.S3_REGION = os.getenv("S3_BUCKET_REGION")
        self.access_key_id = os.getenv("S3_ACCESS_KEY_ID")
        self.secret_access_key = os.getenv("S3_SECRET_KEY")

        # logging.info(f"AWS S3 Configuration:")
        # logging.info(f"  Bucket: {self.bucket_name}")
        # logging.info(f"  Region: {self.S3_REGION}")
        # logging.info(f"  Access Key ID: {self.access_key_id[:10]}...")

        if not all([self.bucket_name, self.S3_REGION, self.access_key_id, self.secret_access_key]):
            raise ValueError("AWS S3 configuration variables must be set.")

        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.S3_REGION
        )

        logging.info(f"AWS S3 client initialized with bucket: {self.bucket_name} (type: {bucket_type.name})")

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage.

        Args:
            filename (str): The filename to sanitize

        Returns:
            str: The sanitized filename
        """
        # Remove or replace problematic characters
        invalid_chars = '<>:"|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')

        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            # Keep extension and truncate name
            max_name_length = 255 - len(ext)
            filename = name[:max_name_length] + ext

        return filename

    def upload_document(self, file_path: str, object_key: Optional[str] = None) -> str:
        """
        Upload a document to AWS S3.

        Args:
            file_path (str): Local path to the file to upload
            object_key (str, optional): S3 object key. If not provided, uses filename from file_path

        Returns:
            str: S3 object key of the uploaded file
        """
        try:
            if not object_key:
                object_key = os.path.basename(file_path)

            with open(file_path, "rb") as file:
                self.s3_client.upload_fileobj(file, self.bucket_name, object_key)

            logging.info(f"File {file_path} uploaded to {object_key}")
            return object_key

        except Exception as e:
            error_msg = f"Failed to upload document to S3: {object_key}. Error: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

    def upload_bytes(self, data: bytes, object_key: str) -> str:
        """
        Upload bytes data to AWS S3.

        Args:
            data (bytes): Bytes data to upload
            object_key (str): S3 object key

        Returns:
            str: S3 object key of the uploaded file
        """
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=data
            )

            return object_key

        except Exception as e:
            error_msg = f"Failed to upload bytes to S3: {object_key}. Error: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

    def upload_text(self, text: str, object_key: str) -> str:
        """
        Upload text data to AWS S3.

        Args:
            text (str): Text data to upload
            object_key (str): S3 object key

        Returns:
            str: S3 object key of the uploaded file
        """
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=text.encode('utf-8')
            )

            logging.info(f"Text data uploaded to {object_key}")
            return object_key

        except Exception as e:
            error_msg = f"Failed to upload text to S3: {object_key}. Error: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

    def create_tender_folder(self, folder_name: str) -> str:
        """
        Creates a logical folder structure for a tender.
        In AWS S3, folders are virtual concepts represented by object key prefixes.

        Args:
            folder_name (str): Unique identifier for the tender

        Returns:
            str: The folder prefix path for the tender
        """
        folder_path = f"tenders/{folder_name}/"

        logging.info(f"Using tender folder structure: {folder_path}")
        return folder_path

    def upload_tender_file(self, folder_name: str, file_type: str, content: Union[str, bytes], file_name: Optional[str] = None) -> str:
        """
        Upload a file to the specific tender folder with standard naming.

        Args:
            folder_name (str): Unique identifier for the tender
            file_type (str): Type of file ('combined_chunks', 'ai_document', 'pdf_document')
            content (str or bytes): Content to upload
            file_name (str, optional): Custom file name to use instead of standard names

        Returns:
            str: S3 object key of the uploaded file
        """
        folder_path = f"tenders/{folder_name}/"

        if not file_name:
            if file_type == 'combined_chunks':
                file_name = 'combined_chunks.json'
            elif file_type == 'ai_document':
                file_name = 'ai_document.md'
            elif file_type == 'pdf_document':
                file_name = 'document.pdf'
            else:
                file_name = f"{file_type}.data"

        object_key = f"{folder_path}{file_name}"

        if isinstance(content, str):
            return self.upload_text(content, object_key)
        elif isinstance(content, bytes):
            self.upload_bytes(content, object_key)
            return object_key
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

    def upload_tender_pdf(self, folder_name: str, pdf_content: bytes, filename: str) -> str:
        """
        Upload a PDF file to the specific tender folder's pdfs directory.

        Args:
            folder_name (str): Unique identifier for the tender
            pdf_content (bytes): PDF content as bytes
            filename (str): Name to use for the PDF file (should include .pdf extension)

        Returns:
            str: S3 object key of the uploaded PDF file
        """
        folder_path = f"tenders/{folder_name}/pdfs/"

        if not filename or filename == 'document.pdf':
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"document_{timestamp}.pdf"

        safe_filename = filename.replace(' ', '_')
        object_key = f"{folder_path}{safe_filename}"

        self.upload_bytes(pdf_content, object_key)

        logging.info(f"PDF uploaded to {object_key}")
        return object_key

    def upload_raw_marker_data(self, folder_name: str, doc_id: str, raw_marker_response: dict) -> str:
        """
        Upload raw marker API response to AWS S3 for logging purposes.

        Args:
            folder_name (str): Unique identifier for the tender
            doc_id (str): Document identifier
            raw_marker_response (dict): Complete raw response from marker API

        Returns:
            str: S3 object key of the uploaded raw marker data
        """
        folder_path = f"tenders/{folder_name}/raw_marker/"
        filename = f"{doc_id}.json"
        object_key = f"{folder_path}{filename}"

        raw_marker_json = json.dumps(raw_marker_response, ensure_ascii=False, indent=2)
        self.upload_text(raw_marker_json, object_key)

        logging.info(f"Raw marker data uploaded to {object_key}")
        return object_key

    def download_document(self, object_key: str, file_path: Optional[str] = None) -> Union[bytes, str]:
        """
        Download a document from AWS S3.

        Args:
            object_key (str): S3 object key
            file_path (str, optional): Local path to save the file. If not provided, returns the content as bytes

        Returns:
            bytes or str: Content as bytes if file_path is not provided, otherwise the path to the saved file
        """
        try:
            if file_path:
                self.s3_client.download_file(self.bucket_name, object_key, file_path)
                logging.info(f"S3 object {object_key} downloaded to {file_path}")
                return file_path
            else:
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=object_key)
                content = response['Body'].read()
                logging.info(f"S3 object {object_key} downloaded as bytes")
                return content

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logging.info(f"S3 object {object_key} not found")
                raise FileNotFoundError(f"S3 object not found: {object_key}")
            else:
                error_msg = f"Failed to download document from S3: {object_key}. Error: {str(e)}"
                logging.error(error_msg)
                raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Failed to download document from S3: {object_key}. Error: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

    def delete_document(self, object_key: str) -> bool:
        """
        Delete a document from AWS S3.

        Args:
            object_key (str): S3 object key

        Returns:
            bool: True if deletion was successful
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=object_key)
            logging.info(f"S3 object {object_key} deleted")
            return True

        except Exception as e:
            error_msg = f"Failed to delete document from S3: {object_key}. Error: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

    def list_documents(self, prefix: Optional[str] = None) -> list:
        """
        List documents in the S3 bucket.

        Args:
            prefix (str, optional): Filter results to items that begin with this prefix

        Returns:
            list: List of S3 object keys
        """
        try:
            kwargs = {'Bucket': self.bucket_name}
            if prefix:
                kwargs['Prefix'] = prefix

            response = self.s3_client.list_objects_v2(**kwargs)

            if 'Contents' not in response:
                return []

            object_keys = [obj['Key'] for obj in response['Contents']]
            return object_keys

        except Exception as e:
            error_msg = f"Failed to list documents in S3 bucket. Error: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

    def object_exists(self, object_key: str) -> bool:
        """
        Check if an object exists in the S3 bucket.

        Args:
            object_key (str): S3 object key

        Returns:
            bool: True if the object exists, False otherwise
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=object_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                raise e

    def list_documents_with_pattern(self, prefix: str, pattern: str) -> list:
        """
        List documents under a prefix that match a pattern.

        Args:
            prefix (str): The prefix to search under
            pattern (str): Pattern to match (e.g., 'chunks.json' to find files ending with chunks.json)

        Returns:
            list: List of matching object keys
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )

            if 'Contents' not in response:
                return []

            matching_files = []
            for obj in response['Contents']:
                key = obj['Key']
                if pattern in key:
                    matching_files.append(key)

            return matching_files
        except Exception as e:
            logging.error(f"Failed to list documents with pattern under {prefix}: {str(e)}")
            return []

    def list_folders(self, prefix: str) -> list:
        """
        List folders (common prefixes) under a given prefix.

        Args:
            prefix (str): The prefix to search under

        Returns:
            list: List of folder names (common prefixes)
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                Delimiter='/'
            )

            if 'CommonPrefixes' not in response:
                return []

            folders = []
            for prefix_info in response['CommonPrefixes']:
                folder_path = prefix_info['Prefix']
                folder_name = folder_path.rstrip('/').split('/')[-1]
                folders.append(folder_name)

            return folders
        except Exception as e:
            logging.error(f"Failed to list folders under {prefix}: {str(e)}")
            return []

    def generate_sas_url(self, object_key: str, minutes: int = 10, read_only: bool = True) -> str:
        """
        Generate a presigned URL for an S3 object.

        Args:
            object_key (str): S3 object key
            minutes (int, optional): Number of minutes the URL will be valid
            read_only (bool, optional): If True, URL will be read-only

        Returns:
            str: Presigned URL for the S3 object
        """
        try:
            method = 'get_object' if read_only else 'put_object'

            presigned_url = self.s3_client.generate_presigned_url(
                method,
                Params={'Bucket': self.bucket_name, 'Key': object_key},
                ExpiresIn=minutes * 60
            )

            logging.info(f"Presigned URL generated for S3 object: {object_key}")
            return presigned_url

        except Exception as e:
            error_msg = f"Failed to generate presigned URL for S3 object: {object_key}. Error: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

    def get_tender_chunks(
        self,
        tender_hash: str,
        chunk_type: Optional[ChunkType] = ChunkType.ALL
    ) -> List[Dict[str, Any]]:
        """
        Retrieves chunks for a specific tender from individual document folders or combined chunks file.

        Args:
            tender_hash: Unique identifier for the tender
            chunk_type: Type of chunks to retrieve (ALL, TECHNICAL, LEGAL, ADDITIONAL, NOTICE)

        Returns:
            List of chunks as dictionaries
        """
        all_chunks = []

        # Step 1: Try to get chunks from individual document type folders
        document_types_to_check = []

        if chunk_type == ChunkType.ALL:
            document_types_to_check = ['technical', 'legal', 'additional', 'notice']
        else:
            document_types_to_check = [chunk_type.value.lower()]

        for doc_type in document_types_to_check:
            prefix = f"tenders/{tender_hash}/{doc_type}/"
            chunk_files = self.list_documents_with_pattern(prefix, "chunks.json")

            for chunk_file in chunk_files:
                try:
                    chunk_content = self.download_document(chunk_file)
                    chunk_json = json.loads(chunk_content)

                    if isinstance(chunk_json, list):
                        all_chunks.extend(chunk_json)
                except Exception as e:
                    logging.warning(f"Error reading chunk file {chunk_file}: {str(e)}")
                    continue

        if all_chunks:
            return all_chunks

        # Step 2: Fallback to combined chunks file in current bucket
        if chunk_type == ChunkType.ALL:
            try:
                chunks_path = f"tenders/{tender_hash}/combined_chunks.json"
                chunks_content = self.download_document(chunks_path)
                chunks_json = json.loads(chunks_content)
                return chunks_json if isinstance(chunks_json, list) else []
            except FileNotFoundError:
                pass
            except Exception as e:
                logging.warning(f"Error reading combined chunks file: {str(e)}")

            # Step 3: Fallback to legacy bucket
            try:
                legacy_client = AWSS3Client(bucket_type=S3BucketType.LEGACY)
                legacy_chunks_path = f"tenders/{tender_hash}/combined_chunks.json"
                legacy_chunks_content = legacy_client.download_document(legacy_chunks_path)
                legacy_chunks_json = json.loads(legacy_chunks_content)
                logging.warning(f"Chunks found in legacy bucket for tender {tender_hash}")
                return legacy_chunks_json if isinstance(legacy_chunks_json, list) else []
            except FileNotFoundError:
                pass
            except Exception as e:
                logging.warning(f"Error reading chunks from legacy bucket: {str(e)}")

        return []
