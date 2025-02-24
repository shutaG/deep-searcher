import ast
from typing import List

from deepsearcher.agent.prompt import get_vector_db_search_prompt

# from deepsearcher.configuration import llm, embedding_model, vector_db
from deepsearcher import configuration
from deepsearcher.tools import log


RERANK_PROMPT = """Based on the query questions and the retrieved chunk, to determine whether the chunk is helpful in answering any of the query question, you can only return "YES" or "NO", without any other information.
Query Questions: {query}
Retrieved Chunk: {retrieved_chunk}

Is the chunk helpful in answering the any of the questions?
"""


async def search_chunks_from_vectordb(query: str, sub_queries: List[str]):
    '''
    1. 获取需要查询的向量库：使用大模型判断问题需要查询的向量库+没有描述的向量库+默认向量库
    2. 从每个向量库查询内容
    3. 让大模型判断和向量库的内容是否和问题有关
    4. 返回与问题有关的的内容列表
    '''


    consume_tokens = 0
    vector_db = configuration.vector_db
    llm = configuration.llm
    embedding_model = configuration.embedding_model
    # query_embedding = embedding_model.embed_query(query)
    collection_infos = vector_db.list_collections()

    # 通过向量库的描述和问题，让大模型选择出需要查询的一个或者多个向量库列表
    vector_db_search_prompt = get_vector_db_search_prompt(
        question=query,
        collection_info=[
            {
                "collection_name": collection_info.collection_name,
                "collection_description": collection_info.description,
            }
            for collection_info in collection_infos
        ],
    )
    chat_response = llm.chat(
        messages=[{"role": "user", "content": vector_db_search_prompt}]
    )
    # 向量库名称的列表
    llm_collections = llm.literal_eval(chat_response.content)
    collection_2_query = {}
    consume_tokens += chat_response.total_tokens

    # llm建议搜索的向量库
    for collection in llm_collections:
        collection_2_query[collection] = query

    # 没有描述的向量库和默认向量库
    for collection_info in collection_infos:
        # If a collection description is not provided, use the query as the search query
        if not collection_info.description:
            collection_2_query[collection_info.collection_name] = query
        # If the default collection exists, use the query as the search query
        if vector_db.default_collection == collection_info.collection_name:
            collection_2_query[collection_info.collection_name] = query
    log.color_print(
        f"<think> Perform search [{query}] on the vector DB collections: {list(collection_2_query.keys())} </think>\n"
    )
    all_retrieved_results = []
    for collection, col_query in collection_2_query.items():
        log.color_print(
            f"<search> Search [{col_query}] in [{collection}]...  </search>\n"
        )
        # 从向量库中搜索当前的问题
        retrieved_results = vector_db.search_data(
            collection=collection, vector=embedding_model.embed_query(col_query)
        )

        # 让大模型判断从向量库中搜索的内容是否和问题（当前问题+所有问题）是否有关联
        accepted_chunk_num = 0
        references = []
        for retrieved_result in retrieved_results:
            chat_response = llm.chat(
                messages=[
                    {
                        "role": "user",
                        "content": RERANK_PROMPT.format(
                            query=[col_query] + sub_queries,
                            retrieved_chunk=retrieved_result.text,
                        ),
                    }
                ]
            )
            consume_tokens += chat_response.total_tokens
            if chat_response.content.startswith("YES"):
                all_retrieved_results.append(retrieved_result)
                accepted_chunk_num += 1
                references.append(retrieved_result.reference)
        if accepted_chunk_num > 0:
            log.color_print(
                f"<search> Accept {accepted_chunk_num} document chunk(s) from references: {references} </search>\n"
            )
    return all_retrieved_results, consume_tokens

    # vector_db.search_data(collection="deepsearcher", vector=query_embedding)
