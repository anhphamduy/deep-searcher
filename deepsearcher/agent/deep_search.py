import asyncio
import json
import os
from typing import List, Tuple

from deepsearcher.agent.base import RAGAgent, describe_class
from deepsearcher.agent.collection_router import CollectionRouter
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.llm.base import BaseLLM
from deepsearcher.tools import log
from deepsearcher.vector_db import RetrievalResult
from deepsearcher.vector_db.base import BaseVectorDB, deduplicate_results

# -----------------------------------------------------------------------
# UPDATED PROMPTS
# -----------------------------------------------------------------------

SUB_QUERY_PROMPT = """To answer this question more comprehensively, please break down the original question into up to five sub-questions. Return as list of str.
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

RERANK_PROMPT = """Based on the query questions and the retrieved chunk, determine whether the chunk is helpful in answering any of the query questions. You can only return "YES" or "NO" or "MAYBE", without any other information.

Query Questions: {query}
Retrieved Chunk: {retrieved_chunk}

Is the chunk helpful in answering any of the questions?
"""

########################################################################
# MAIN CHANGE: REFLECT_PROMPT returning JSON with "reason" and "questions"
########################################################################
REFLECT_PROMPT = """Determine whether additional search queries are needed based on the original query, previous sub queries, and all retrieved document chunks.
Output valid JSON with two fields: "reason" (a string) and "questions" (an array of strings).

- "reason" should explain why more questions are (or aren't) needed.
- "questions" is a list of at most 4 additional search questions. If no further research is needed, keep "questions" empty.

Example valid JSON:
{{
  "reason": "We need more information about prior studies on the subject.",
  "questions": ["What prior research is cited?", "Are there any references to external data?"]
}}

If no further research is needed, return an empty list for "questions" with an explanation in "reason".

Original Query: {question}
Previous Sub Queries: {mini_questions}

Related Chunks: 
{mini_chunk_str}
"""

SUMMARY_PROMPT = """You are an AI content analysis expert. Please write a concise summary of all relevant chunks below. 
Preserve key facts, references, or citations if present. Output plain text.

Original Query:
{question}

Sub-Queries:
{mini_questions}

Relevant Document Chunks:
{mini_chunk_str}
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
            self.collection_router = CollectionRouter(
                llm=self.llm, vector_db=self.vector_db
            )
        self.text_window_splitter = text_window_splitter

    def _generate_sub_queries(self, original_query: str) -> Tuple[List[str], int]:
        """Given the original query, generate up to four sub-queries (or a single one if trivial)."""
        chat_response = self.llm.chat(
            messages=[
                {
                    "role": "user",
                    "content": SUB_QUERY_PROMPT.format(original_query=original_query),
                }
            ]
        )
        response_content = chat_response.content
        return self.llm.literal_eval(response_content), chat_response.total_tokens

    async def _search_chunks_from_vectordb(
        self,
        query: str,
        sub_queries: List[str],
        thinking_callback,
        question_id: str,
        session_id,
    ) -> Tuple[List[RetrievalResult], int]:
        """Search the vector DB for the given query and sub-queries. Return accepted chunks plus token usage."""
        consume_tokens = 0

        # 1) Determine which sources to search in
        if self.route_collection:
            selected_collections, n_token_route = self.collection_router.invoke(
                query=query
            )
        else:
            selected_collections = self.collection_router.all_collections
            n_token_route = 0
        consume_tokens += n_token_route

        all_retrieved_results = []
        query_vector = self.embedding_model.embed_query(query)

        # 2) Search each source
        for source in selected_collections:
            retrieved_results = await self.vector_db.asearch_data(
                collection=source, vector=query_vector, session_id=session_id
            )
            if not retrieved_results:
                log.color_print(f"😕 No snippets found in {source}.\n")
                continue

            # 3) Rerank each snippet concurrently
            tasks_rerank = []
            for result in retrieved_results:
                snippet = result.text[:80].replace("\n", " ")
                if len(result.text) > 80:
                    snippet += "..."

                async def rerank_snippet(retrieved_result=result, snippet=snippet):
                    chat_response = await self.llm.achat(
                        messages=[
                            {
                                "role": "user",
                                "content": RERANK_PROMPT.format(
                                    query=[query] + sub_queries,
                                    retrieved_chunk=f"<chunk>{retrieved_result.text}</chunk>",
                                ),
                            }
                        ]
                    )
                    response_content = chat_response.content.strip()

                    # Remove hidden reasoning if present
                    if "<think>" in response_content and "</think>" in response_content:
                        end_of_think = response_content.find("</think>") + len(
                            "</think>"
                        )
                        response_content = response_content[end_of_think:].strip()

                    return (
                        retrieved_result,
                        response_content,
                        chat_response.total_tokens,
                    )

                tasks_rerank.append(rerank_snippet())

            rerank_outcomes = await asyncio.gather(*tasks_rerank)

            chunk_eval_event = {
                "eventType": "chunk-evaluation",
                "questionId": question_id,
                "question": query,
                "chunks": [],
            }

            accepted_count = 0
            references = set()
            for retrieved_result, response_content, used_tokens in rerank_outcomes:
                consume_tokens += used_tokens
                filename = os.path.basename(retrieved_result.reference)

                # If it says "YES" or "MAYBE", accept the snippet
                if (
                    "YES" in response_content or "MAYBE" in response_content
                ) and "NO" not in response_content:
                    chunk_eval_event["chunks"].append(
                        {
                            "chunk": retrieved_result.text,
                            "evaluation": response_content,
                            "filename": filename,
                        }
                    )
                    all_retrieved_results.append(retrieved_result)
                    accepted_count += 1
                    references.add(retrieved_result.reference)

            # Emit chunk-evaluation event
            thinking_callback(chunk_eval_event)

            if accepted_count > 0:
                link_list = []
                for ref in references:
                    file_name = os.path.basename(ref)
                    link_list.append(f"[{file_name}]")
                references_text = ", ".join(link_list)
                msg = (
                    f'✔️ Found {accepted_count} helpful snippet(s) for "{query}".\n'
                    f"Relevant files: {references_text}"
                )
            else:
                log.color_print("🙁 None of these snippets seemed helpful.\n")

        return all_retrieved_results, consume_tokens

    ########################################################################
    # UPDATED GAP QUERIES: parse JSON -> { "reason": "...", "questions": [] }
    ########################################################################
    def _generate_gap_queries(
        self,
        original_query: str,
        all_sub_queries: List[str],
        all_chunks: List[RetrievalResult],
    ) -> Tuple[str, List[str], int]:
        """Reflect to see if additional queries are needed to fill knowledge gaps.
        Returns (reason, questions, tokens_used).
        If 'questions' is empty => final answer scenario."""
        if len(all_chunks) > 0:
            mini_chunk_str = self._format_chunk_texts(
                [chunk.text for chunk in all_chunks]
            )
        else:
            mini_chunk_str = "NO RELATED CHUNKS FOUND."

        reflect_prompt = REFLECT_PROMPT.format(
            question=original_query,
            mini_questions=all_sub_queries,
            mini_chunk_str=mini_chunk_str,
        )
        chat_response = self.llm.chat(
            [{"role": "user", "content": reflect_prompt}], json_mode=True
        )
        response_content = chat_response.content
        tokens_used = chat_response.total_tokens

        # Now parse the JSON.
        # We expect something like:
        # {
        #   "reason": "...",
        #   "questions": ["...", "..."]
        # }
        try:
            reflect_result = json.loads(response_content)
            reason = reflect_result.get("reason", "")
            questions = reflect_result.get("questions", [])
        except json.JSONDecodeError:
            # Fallback if the model returned something else
            reason = "Could not parse reflection JSON."
            questions = []

        return reason, questions, tokens_used

    ########################################################################
    # Additional helper to produce final chunk summary with SUMMARY_PROMPT
    ########################################################################
    def _generate_summary(
        self,
        original_query: str,
        all_sub_queries: List[str],
        all_chunks: List[RetrievalResult],
    ) -> Tuple[str, int]:
        """Use SUMMARY_PROMPT to create a short final summary from all retrieved chunks."""
        if not all_chunks:
            return "No relevant information found.", 0

        # Format chunk strings
        chunk_texts = [chunk.text for chunk in all_chunks]
        chunk_str = self._format_chunk_texts(chunk_texts)

        prompt_text = SUMMARY_PROMPT.format(
            question=original_query,
            mini_questions=all_sub_queries,
            mini_chunk_str=chunk_str,
        )
        chat_response = self.llm.chat([{"role": "user", "content": prompt_text}])
        summary_text = chat_response.content
        tokens_used = chat_response.total_tokens

        return summary_text, tokens_used

    def retrieve(
        self, original_query: str, **kwargs
    ) -> Tuple[List[RetrievalResult], int, dict]:
        """Convenient sync wrapper for async_retrieve."""
        return asyncio.run(self.async_retrieve(original_query, **kwargs))

    async def async_retrieve(
        self, original_query: str, **kwargs
    ) -> Tuple[List[RetrievalResult], int, dict]:
        """
        Orchestrate the search with sub-queries and reflection for additional queries.
        - Calls reflection which now returns JSON with { reason, questions }.
        - If questions are empty => final answer scenario.
        """
        max_iter = kwargs.pop("max_iter", self.max_iter)
        thinking_callback = kwargs.get("thinking_callback", lambda x: None)

        log.color_print(f"<query> {original_query} </query>\n")

        all_search_res = []
        all_sub_queries = []
        total_tokens = 0

        # 1) Generate sub-queries
        sub_queries, used_token = self._generate_sub_queries(original_query)
        total_tokens += used_token

        # Let caller know which sub-questions we ended up with
        thinking_callback(
            {
                "eventType": "questions-generated",
                "questions": sub_queries,
            }
        )

        if not sub_queries:
            # No sub-queries means we can produce a final answer right away
            # But let's say no relevant info
            thinking_callback(
                {
                    "event": "message",
                    "data": {
                        "eventType": "final-answer",
                        "researchSessionId": "67890",
                        "answer": f"No sub-queries needed. Possibly no relevant info for '{original_query}'",
                        "relevant_chunks": [],
                    },
                }
            )
            return [], total_tokens, {}

        all_sub_queries.extend(sub_queries)

        # 2) Iterative retrieval
        question_counter = 1
        for iteration in range(max_iter):
            log.color_print(f">> Iteration: {iteration + 1}\n")
            sub_gap_queries = sub_queries

            # We'll search each sub-gap query in parallel
            search_tasks = []
            for i, sq in enumerate(sub_gap_queries, start=1):
                qid = f"q{question_counter}"
                question_counter += 1
                search_tasks.append(
                    self._search_chunks_from_vectordb(
                        sq,
                        sub_gap_queries,
                        thinking_callback,
                        question_id=qid,
                        session_id=kwargs['file_index_session_id']
                    )
                )

            # Wait for all parallel searches
            search_results = await asyncio.gather(*search_tasks)

            # Merge all results
            search_res_from_vectordb = []
            for res, consumed_token in search_results:
                total_tokens += consumed_token
                search_res_from_vectordb.extend(res)

            # Deduplicate
            search_res_from_vectordb = deduplicate_results(search_res_from_vectordb)
            all_search_res.extend(search_res_from_vectordb)
            all_search_res = deduplicate_results(all_search_res)

            # 3) Reflection for gap queries
            if iteration == max_iter - 1:
                log.color_print("<think> Exceeded max iterations. </think>\n")
                # We'll just break and finalize
                break

            reason, new_questions, consumed_token = self._generate_gap_queries(
                original_query, all_sub_queries, all_search_res
            )
            total_tokens += consumed_token

            # If new_questions is not empty => we have another iteration
            if new_questions:                
                thinking_callback(
                    {
                        "event": "message",
                        "data": {
                            "eventType": "reflection",
                            "researchSessionId": "67890",
                            "step": 2,
                            "reflection": reason or "Additional search needed.",
                        },
                    }
                )
                thinking_callback(
                    {
                        "eventType": "questions-generated",
                        "questions": new_questions,
                    }
                )
                
                all_sub_queries.extend(new_questions)
                sub_queries = new_questions
            else:
                # No new questions => final answer scenario
                thinking_callback(
                    {
                        "event": "message",
                        "data": {
                            "eventType": "reflection",
                            "step": iteration,
                            "reflection": reason or "No further queries needed.",
                        },
                    }
                )
                break

        # If we exit the loop, let's produce a final answer event:
        # 4) Produce final summary
        summary_text, sum_tokens = self._generate_summary(
            original_query, all_sub_queries, all_search_res
        )
        total_tokens += sum_tokens

        # Format relevant chunks to the requested final structure
        relevant_chunks_data = []
        for r in all_search_res:
            relevant_chunks_data.append(
                {"file_path": r.reference, "relevant_content": r.text}
            )

        thinking_callback(
            {
                "event": "message",
                "data": {
                    "eventType": "final-answer",
                    "researchSessionId": "67890",
                    "answer": summary_text,
                    "relevant_chunks": relevant_chunks_data,
                },
            }
        )

        additional_info = {"all_sub_queries": all_sub_queries}
        return all_search_res, total_tokens, additional_info

    def query(self, query: str, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
        """
        1) Retrieves relevant chunks for 'query'.
        2) Summarizes them and returns final answer + retrieval results.
        """
        all_retrieved_results, n_token_retrieval, additional_info = self.retrieve(
            query, **kwargs
        )
        if not all_retrieved_results:
            return (
                f"No relevant information found for query '{query}'.",
                [],
                n_token_retrieval,
            )

        # Return "answer" plus the raw retrieval results
        # (the final answer was already sent to thinking_callback event above)
        return (
            "See final-answer event above for summary.",
            all_retrieved_results,
            n_token_retrieval,
        )

    #######################################################################
    # Helper to format chunk texts with <chunk_i> ...
    #######################################################################
    def _format_chunk_texts(self, chunk_texts: List[str]) -> str:
        chunk_str = ""
        for i, chunk in enumerate(chunk_texts):
            chunk_str += f"<chunk_{i}>\n{chunk}\n</chunk_{i}>\n"
        return chunk_str
