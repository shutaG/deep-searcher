import os
import shutil
from typing import List

from langchain_core.documents import Document

from deepsearcher.loader.file_loader.base import BaseLoader
from deepsearcher.tools import log


class UnstructuredLoader(BaseLoader):
    """
    Loader for unstructured documents using the unstructured-io library.

    This loader processes various document formats using the unstructured-io library's
    processing pipeline, extracting text and metadata from complex document formats.
    """

    def __init__(self):
        """
        Initialize the UnstructuredLoader.

        Creates a temporary directory for processed outputs and cleans up any existing ones.
        """
        self.directory_with_results = "./pdf_processed_outputs"
        if os.path.exists(self.directory_with_results):
            shutil.rmtree(self.directory_with_results)
        os.makedirs(self.directory_with_results)

    def load_pipeline(self, input_path: str) -> List[Document]:
        """
        Process documents using the unstructured-io pipeline.

        Args:
            input_path: Path to the file or directory to be processed.

        Returns:
            A list of Document objects extracted from the processed files.

        Note:
            Requires environment variables UNSTRUCTURED_API_KEY and UNSTRUCTURED_API_URL to be set.
        """
        from unstructured_ingest.v2.interfaces import ProcessorConfig
        from unstructured_ingest.v2.pipeline.pipeline import Pipeline
        from unstructured_ingest.v2.processes.connectors.local import (
            LocalConnectionConfig,
            LocalDownloaderConfig,
            LocalIndexerConfig,
            LocalUploaderConfig,
        )
        from unstructured_ingest.v2.processes.partitioner import PartitionerConfig

        Pipeline.from_configs(
            context=ProcessorConfig(),
            indexer_config=LocalIndexerConfig(input_path=input_path),
            downloader_config=LocalDownloaderConfig(),
            source_connection_config=LocalConnectionConfig(),
            partitioner_config=PartitionerConfig(
                partition_by_api=True,
                api_key=os.getenv("UNSTRUCTURED_API_KEY"),
                partition_endpoint=os.getenv("UNSTRUCTURED_API_URL"),
                strategy="hi_res",
                additional_partition_args={
                    "split_pdf_page": True,
                    "split_pdf_concurrency_level": 15,
                },
            ),
            uploader_config=LocalUploaderConfig(output_dir=self.directory_with_results),
        ).run()

        from unstructured.staging.base import elements_from_json

        elements = []
        for filename in os.listdir(self.directory_with_results):
            if filename.endswith(".json"):
                file_path = os.path.join(self.directory_with_results, filename)
                try:
                    elements.extend(elements_from_json(filename=file_path))
                except IOError:
                    log.color_print(f"Error: Could not read file {filename}.")

        documents = []
        for element in elements:
            metadata = element.metadata.to_dict()
            metadata["reference"] = input_path  # TODO test it
            documents.append(
                Document(
                    page_content=element.text,
                    metadata=metadata,
                )
            )
        return documents

    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a single file using the unstructured-io pipeline.

        Args:
            file_path: Path to the file to be processed.

        Returns:
            A list of Document objects extracted from the processed file.
        """
        return self.load_pipeline(file_path)

    def load_directory(self, directory: str) -> List[Document]:
        """
        Load all supported files from a directory using the unstructured-io pipeline.

        Args:
            directory: Path to the directory containing files to be processed.

        Returns:
            A list of Document objects extracted from all processed files.
        """
        return self.load_pipeline(directory)

    @property
    def supported_file_types(self) -> List[str]:  # TODO
        """
        Get the list of file extensions supported by this loader.

        Returns:
            A list of supported file extensions: ["pdf"].

        Note:
            This is currently limited to PDF files, but the unstructured-io library
            supports many more formats that could be added in the future.
        """
        return ["pdf"]
