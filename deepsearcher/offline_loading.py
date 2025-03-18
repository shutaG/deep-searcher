import os
from typing import List, Union

from tqdm import tqdm

# from deepsearcher.configuration import embedding_model, vector_db, file_loader
from deepsearcher import configuration
from deepsearcher.loader.splitter import split_docs_to_chunks


def load_from_local_files(
    paths_or_directory: Union[str, List[str]],
    collection_name: str = None,
    collection_description: str = None,
    force_new_collection: bool = False,
    chunk_size: int = 1500,
    chunk_overlap: int = 100,
    batch_size: int = 256,
):
    """
    Load knowledge from local files or directories into the vector database.

    This function processes files from the specified paths or directories,
    splits them into chunks, embeds the chunks, and stores them in the vector database.

    Args:
        paths_or_directory: A single path or a list of paths to files or directories to load.
        collection_name: Name of the collection to store the data in. If None, uses the default collection.
        collection_description: Description of the collection. If None, no description is set.
        force_new_collection: If True, drops the existing collection and creates a new one.
        chunk_size: Size of each chunk in characters.
        chunk_overlap: Number of characters to overlap between chunks.
        batch_size: Number of chunks to process at once during embedding.

    Raises:
        FileNotFoundError: If any of the specified paths do not exist.
    """
    # 初始化向量数据库
    vector_db = configuration.vector_db
    if collection_name is None:
        collection_name = vector_db.default_collection
    collection_name = collection_name.replace(" ", "_").replace("-", "_")
    embedding_model = configuration.embedding_model
    file_loader = configuration.file_loader
    vector_db.init_collection(
        dim=embedding_model.dimension,
        collection=collection_name,
        description=collection_description,
        force_new_collection=force_new_collection,
    )

    # 获取文件路径或者文件夹下所有支持的文件类型的路径
    if isinstance(paths_or_directory, str):
        paths_or_directory = [paths_or_directory]
    all_docs = []
    for path in tqdm(paths_or_directory, desc="Loading files"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: File or directory '{path}' does not exist.")
        if os.path.isdir(path):
            docs = file_loader.load_directory(path)
        else:
            docs = file_loader.load_file(path)
        all_docs.extend(docs)
    # print("Splitting docs to chunks...")
    # 将文件切分为chunk列表
    chunks = split_docs_to_chunks(
        all_docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    # 进行embedding
    chunks = embedding_model.embed_chunks(chunks, batch_size=batch_size)
    # 插入向量数据库
    vector_db.insert_data(collection=collection_name, chunks=chunks)


def load_from_website(
    urls: Union[str, List[str]],
    collection_name: str = None,
    collection_description: str = None,
    force_new_collection: bool = False,
    batch_size: int = 256,
    **crawl_kwargs,
):
    """
    Load knowledge from websites into the vector database.

    This function crawls the specified URLs, processes the content,
    splits it into chunks, embeds the chunks, and stores them in the vector database.

    Args:
        urls: A single URL or a list of URLs to crawl.
        collection_name: Name of the collection to store the data in. If None, uses the default collection.
        collection_description: Description of the collection. If None, no description is set.
        force_new_collection: If True, drops the existing collection and creates a new one.
        batch_size: Number of chunks to process at once during embedding.
        **crawl_kwargs: Additional keyword arguments to pass to the web crawler.
    """
    '''
    逐个从网址读取内容，并加载到向量库
    '''
    if isinstance(urls, str):
        urls = [urls]
    vector_db = configuration.vector_db
    embedding_model = configuration.embedding_model
    web_crawler = configuration.web_crawler

    vector_db.init_collection(
        dim=embedding_model.dimension,
        collection=collection_name,
        description=collection_description,
        force_new_collection=force_new_collection,
    )

    all_docs = web_crawler.crawl_urls(urls, **crawl_kwargs)
    # 切分网页内容
    chunks = split_docs_to_chunks(all_docs)
    chunks = embedding_model.embed_chunks(chunks, batch_size=batch_size)
    vector_db.insert_data(collection=collection_name, chunks=chunks)
