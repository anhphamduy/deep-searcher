import asyncio
from typing import List, Tuple

from deepsearcher.agent.base import RAGAgent, describe_class
from deepsearcher.agent.collection_router import CollectionRouter
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.llm.base import BaseLLM
from deepsearcher.tools import log
from deepsearcher.vector_db import RetrievalResult
from deepsearcher.vector_db.base import BaseVectorDB, deduplicate_results

SUB_QUERY_PROMPT = """To answer this question more comprehensively, please break down the original question into up to four sub-questions. Return as list of str.
If this is a very simple question and no decomposition is necessary, then keep the only one original question in the python code list.

Original Question: {original_query}


<EXAMPLE>
Example input:
"Explain deep learning"

Example output:
[
    "What is deep learning?",
    "What is the difference between deep learning and machine learning?",
    "What is the history of deep learning?"
]
</EXAMPLE>

Provide your response in a python code list of str format:
"""

RERANK_PROMPT = """Based on the query questions and the retrieved chunk, to determine whether the chunk is helpful in answering any of the query question, you can only return "YES" or "NO" or "MAYBE", without any other information.

Query Questions: {query}
Retrieved Chunk: {retrieved_chunk}

Is the chunk helpful in answering any of the questions?
"""

REFLECT_PROMPT = """Determine whether additional search queries are needed based on the original query, previous sub queries, and all retrieved document chunks. If further research is required, provide a Python list of up to 3 search queries. If no further research is required, return an empty list. Further search is required when there is a lack of details, or more clarifications or definitions are needed, or if the search results are too generalized.

If the original query is to write a report, then you prefer to generate some further queries, otherwise return an empty list.

Original Query: {question}

Previous Sub Queries: {mini_questions}

Related Chunks: 
{mini_chunk_str}

Respond exclusively in valid List of str format without any other text.
"""

SUMMARY_PROMPT = """You are an AI content analysis expert with strong skills in restructuring and refining content for better comprehension. Your task is to rewrite the provided information into a well-structured, coherent, and AI-friendly format while preserving all details, including the smallest details. Avoid over-generalization.

Original Query:
{question}

Previous Sub-Queries:
{mini_questions}

Relevant Document Chunks:
{mini_chunk_str}
"""

# -----------------------------------------------------------------------
# UPDATED PROMPTS: We pass the question, sub-queries, and chunk texts too
# -----------------------------------------------------------------------
REVIEW_PROMPT = """Review the following rewrite for correctness, completeness, and clarity. 
We are also including the original query, sub-queries, and relevant chunks. 
Use them to verify or correct missing details or too generalized details.
It must be as explicit and detailed as possible.

Original Query:
{question}

Sub-Queries:
{sub_queries}

Chunks:
{chunks}

Rewrite to review:
{summarization}
"""

DETAILED_REWRITE_PROMPT = """Now rewrite the first rewrite to be more detailed and thorough, incorporating any missing definitions, clarifications, or important details referenced in the question, sub-queries, or chunks. 
Return the final detailed summary as plain text.

Original Query:
{question}

Sub-Queries:
{sub_queries}

Chunks:
{chunks}

Feedback:
{improved_summarization}
"""

@describe_class(
    "This agent is suitable for handling general and simple queries, such as writing a report, survey, or article."
)
class DeepSearch(RAGAgent):
    def __init__(
        self,
        llm: BaseLLM,
        embedding_model: BaseEmbedding,
        vector_db: BaseVectorDB,
        max_iter: int = 3,
        route_collection: bool = True,
        text_window_splitter: bool = True,
        **kwargs,
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.max_iter = max_iter
        self.route_collection = route_collection
        if self.route_collection:
            self.collection_router = CollectionRouter(llm=self.llm, vector_db=self.vector_db)
        self.text_window_splitter = text_window_splitter

    def _generate_sub_queries(self, original_query: str) -> Tuple[List[str], int]:
        """Given the original query, generate up to four sub-queries (or a single one if trivial)."""
        chat_response = self.llm.chat(
            messages=[
                {"role": "user", "content": SUB_QUERY_PROMPT.format(original_query=original_query)}
            ]
        )
        response_content = chat_response.content
        return self.llm.literal_eval(response_content), chat_response.total_tokens

    async def _search_chunks_from_vectordb(
        self, query: str, sub_queries: List[str], thinking_callback
    ) -> Tuple[List[RetrievalResult], int]:
        """Search the vector DB for the given query and sub-queries. Return accepted chunks plus token usage."""
        consume_tokens = 0

        # 1) Determine which collections to search in
        if self.route_collection:
            selected_collections, n_token_route = self.collection_router.invoke(query=query)
        else:
            selected_collections = self.collection_router.all_collections
            n_token_route = 0
        consume_tokens += n_token_route

        all_retrieved_results = []
        query_vector = self.embedding_model.embed_query(query)

        # 2) Search each collection
        for collection in selected_collections:
            log.color_print(f"<search> Searching for [{query}] in [{collection}]... </search>\n")
            thinking_callback(f"<search> Searching for [{query}] in [{collection}]... </search>\n")
            retrieved_results = self.vector_db.search_data(
                collection=collection,
                vector=query_vector
            )
            if not retrieved_results:
                log.color_print(
                    f"<search> No relevant doc chunks found in '{collection}'. </search>\n"
                )
                continue

            # 3) Rerank each chunk to confirm helpfulness
            accepted_chunk_num = 0
            references = set()

            for retrieved_result in retrieved_results:
                chat_response = self.llm.chat(
                    messages=[
                        {
                            "role": "user",
                            "content": RERANK_PROMPT.format(
                                query=[query] + sub_queries,
                                retrieved_chunk=f"<chunk>{retrieved_result.text}</chunk>"
                            ),
                        }
                    ]
                )
                consume_tokens += chat_response.total_tokens
                response_content = chat_response.content.strip()

                # Remove hidden reasoning if present
                if "<think>" in response_content and "</think>" in response_content:
                    end_of_think = response_content.find("</think>") + len("</think>")
                    response_content = response_content[end_of_think:].strip()

                # If it says "YES" or "MAYBE", accept the chunk
                if ("YES" in response_content or "MAYBE" in response_content) and "NO" not in response_content:
                    all_retrieved_results.append(retrieved_result)
                    accepted_chunk_num += 1
                    references.add(retrieved_result.reference)

            if accepted_chunk_num > 0:
                thinking_callback(f"<search> Accepted {accepted_chunk_num} chunk(s) from references: {list(references)} </search>\n")
                log.color_print(
                    f"<search> Accepted {accepted_chunk_num} chunk(s) from references: {list(references)} </search>\n"
                )
            else:
                log.color_print(
                    f"<search> No chunk accepted from '{collection}'. </search>\n"
                )

        return all_retrieved_results, consume_tokens

    def _generate_gap_queries(
        self,
        original_query: str,
        all_sub_queries: List[str],
        all_chunks: List[RetrievalResult]
    ) -> Tuple[List[str], int]:
        """Reflect to see if additional queries are needed to fill knowledge gaps."""
        if len(all_chunks) > 0:
            mini_chunk_str = self._format_chunk_texts([chunk.text for chunk in all_chunks])
        else:
            mini_chunk_str = "NO RELATED CHUNKS FOUND."

        reflect_prompt = REFLECT_PROMPT.format(
            question=original_query,
            mini_questions=all_sub_queries,
            mini_chunk_str=mini_chunk_str
        )
        chat_response = self.llm.chat([{"role": "user", "content": reflect_prompt}])
        response_content = chat_response.content
        return self.llm.literal_eval(response_content), chat_response.total_tokens

    def retrieve(
        self, original_query: str, **kwargs
    ) -> Tuple[List[RetrievalResult], int, dict]:
        """Convenient sync wrapper for async_retrieve."""
        return asyncio.run(self.async_retrieve(original_query, **kwargs))

    async def async_retrieve(
        self, original_query: str, **kwargs
    ) -> Tuple[List[RetrievalResult], int, dict]:
        """Orchestrate the search with sub-queries and reflection for additional queries."""
        max_iter = kwargs.pop("max_iter", self.max_iter)
        log.color_print(f"<query> {original_query} </query>\n")

        all_search_res = []
        all_sub_queries = []
        total_tokens = 0

        # 1) Generate sub-queries
        sub_queries, used_token = self._generate_sub_queries(original_query)
        total_tokens += used_token
        if not sub_queries:
            log.color_print("<think> No sub-queries generated. Exiting. </think>\n")
            return [], total_tokens, {}

        log.color_print(
            f"<think> Sub-queries for '{original_query}': {sub_queries}</think>\n"
        )
        all_sub_queries.extend(sub_queries)
        sub_gap_queries = sub_queries

        # 2) Iterative retrieval
        for iteration in range(max_iter):
            log.color_print(f">> Iteration: {iteration + 1}\n")
            search_res_from_vectordb = []
            search_res_from_internet = []  # (Placeholder)

            # Create tasks (parallel searches for each sub-gap query)
            search_tasks = [
                self._search_chunks_from_vectordb(q, sub_gap_queries, kwargs['thinking_callback'])
                for q in sub_gap_queries
            ]
            search_results = await asyncio.gather(*search_tasks)

            # Merge all results
            for res, consumed_token in search_results:
                total_tokens += consumed_token
                search_res_from_vectordb.extend(res)

            # Deduplicate
            search_res_from_vectordb = deduplicate_results(search_res_from_vectordb)
            all_search_res.extend(search_res_from_vectordb + search_res_from_internet)

            if iteration == max_iter - 1:
                log.color_print("<think> Exceeded max iterations. </think>\n")
                break

            # 3) Reflection for gap queries
            log.color_print("<think> Reflecting on search results... </think>\n")
            kwargs['thinking_callback']("Reflecting on the search results...")
            sub_gap_queries, consumed_token = self._generate_gap_queries(
                original_query, all_sub_queries, all_search_res
            )
            total_tokens += consumed_token

            if not sub_gap_queries:
                log.color_print("<think> No new queries generated. Exiting. </think>\n")
                break

            log.color_print(
                f"<think> Additional queries: {sub_gap_queries} </think>\n"
            )
            all_sub_queries.extend(sub_gap_queries)

        # Final deduplication
        all_search_res = deduplicate_results(all_search_res)
        additional_info = {"all_sub_queries": all_sub_queries}
        return all_search_res, total_tokens, additional_info

    def query(self, query: str, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
        """
        1) Retrieves relevant chunks for 'query'.
        2) Summarizes them using SUMMARY_PROMPT.
        3) Reviews the summary using REVIEW_PROMPT (with question/sub-queries/chunks).
        4) Rewrites the improved summary in more detail using DETAILED_REWRITE_PROMPT.
        Returns the final answer and retrieval results.
        """
        # -- 1) Retrieve
        all_retrieved_results, n_token_retrieval, additional_info = self.retrieve(query, **kwargs)
        if not all_retrieved_results:
            return f"No relevant information found for query '{query}'.", [], n_token_retrieval

        all_sub_queries = additional_info["all_sub_queries"]
        # Decide which text to collect from each chunk
        chunk_texts = []
        for chunk in all_retrieved_results:
            if self.text_window_splitter and "wider_text" in chunk.metadata:
                chunk_texts.append(chunk.metadata["wider_text"])
            else:
                chunk_texts.append(chunk.text)

        # -- 2) Summarize with SUMMARY_PROMPT
        log.color_print(f"<think> Summarizing from {len(all_retrieved_results)} chunks... </think>\n")
        kwargs['thinking_callback']("Writting the final results")
        summary_prompt = SUMMARY_PROMPT.format(
            question=query,
            mini_questions=all_sub_queries,
            mini_chunk_str=self._format_chunk_texts(chunk_texts),
        )
        chat_response_summary = self.llm.chat(
            [{"role": "user", "content": summary_prompt}]
        )
        summary_text = chat_response_summary.content

        # -- 3) Review the summary with REVIEW_PROMPT
        chunks_str = self._format_chunk_texts(chunk_texts)
        review_prompt = REVIEW_PROMPT.format(
            question=query,
            sub_queries=all_sub_queries,
            chunks=chunks_str,
            summarization=summary_text
        )
        chat_response_review = self.llm.chat(
            [{"role": "user", "content": review_prompt}]
        )
        improved_summarization = chat_response_review.content

        # -- 4) Rewrite in detail with DETAILED_REWRITE_PROMPT
        rewrite_prompt = DETAILED_REWRITE_PROMPT.format(
            question=query,
            sub_queries=all_sub_queries,
            chunks=chunks_str,
            improved_summarization=improved_summarization
        )
        chat_response_detailed = self.llm.chat(
            [{"role": "user", "content": rewrite_prompt}]
        )
        final_answer = chat_response_detailed.content

        # Print final answer to logs
        log.color_print("\n==== FINAL ANSWER====\n")
        log.color_print(final_answer)

        # Tally up total tokens
        total_tokens = (
            n_token_retrieval
            + chat_response_summary.total_tokens
            + chat_response_review.total_tokens
            + chat_response_detailed.total_tokens
        )

        return final_answer, all_retrieved_results, total_tokens

    def _format_chunk_texts(self, chunk_texts: List[str]) -> str:
        """Conveniently format chunk texts for prompts."""
        chunk_str = ""
        for i, chunk in enumerate(chunk_texts):
            chunk_str += f"<chunk_{i}>\n{chunk}\n</chunk_{i}>\n"
        return chunk_str
