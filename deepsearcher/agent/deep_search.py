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
        try:
            chat_response = self.llm.chat(
                messages=[
                    {
                        "role": "user",
                        "content": SUB_QUERY_PROMPT.format(original_query=original_query),
                    }
                ]
            )
            response_content = chat_response.content
            sub_queries = self.llm.literal_eval(response_content)
            return sub_queries, chat_response.total_tokens
        except Exception as e:
            # If we can't generate sub-queries, return the original query as fallback
            log.color_print(f"⚠️ Failed to generate sub-queries: {str(e)}\n")
            return [original_query], 0

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
        try:
            if self.route_collection:
                selected_collections, n_token_route = self.collection_router.invoke(
                    query=query
                )
            else:
                selected_collections = self.collection_router.all_collections
                n_token_route = 0
            consume_tokens += n_token_route
        except Exception as e:
            await thinking_callback({
                "eventType": "research-failures",
                "subEventType": "collection-routing-failure",
                "error": str(e),
                "details": {
                    "query": query,
                    "questionId": question_id
                }
            })
            # Fallback to all collections if routing fails
            selected_collections = getattr(self.collection_router, 'all_collections', [])
            if not selected_collections:
                return [], consume_tokens

        all_retrieved_results = []
        
        try:
            query_vector = self.embedding_model.embed_query(query)
        except Exception as e:
            await thinking_callback({
                "eventType": "research-failures",
                "subEventType": "embedding-failure",
                "error": str(e),
                "details": {
                    "query": query,
                    "questionId": question_id
                }
            })
            return [], consume_tokens

        # 2) Search each source
        for source in selected_collections:
            try:
                retrieved_results = await self.vector_db.asearch_data(
                    collection=source, vector=query_vector, session_id=session_id
                )
            except Exception as e:
                await thinking_callback({
                    "eventType": "research-failures",
                    "subEventType": "vector-db-search-failure",
                    "error": str(e),
                    "details": {
                        "query": query,
                        "collection": source,
                        "questionId": question_id
                    }
                })
                continue
                
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
                    try:
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
                    except Exception as e:
                        # Return a conservative evaluation on reranking failure
                        return (retrieved_result, "MAYBE", 0)

                tasks_rerank.append(rerank_snippet())

            rerank_outcomes = await asyncio.gather(*tasks_rerank, return_exceptions=True)

            chunk_eval_event = {
                "eventType": "chunk-evaluation",
                "questionId": question_id,
                "question": query,
                "chunks": [],
            }

            accepted_count = 0
            references = set()
            rerank_failures = 0
            
            for outcome in rerank_outcomes:
                if isinstance(outcome, Exception):
                    rerank_failures += 1
                    continue
                    
                retrieved_result, response_content, used_tokens = outcome
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

            # Report reranking failures if any
            if rerank_failures > 0:
                await thinking_callback({
                    "eventType": "research-failures",
                    "subEventType": "reranking-partial-failure",
                    "error": f"Failed to rerank {rerank_failures} chunks",
                    "details": {
                        "query": query,
                        "collection": source,
                        "questionId": question_id,
                        "failedCount": rerank_failures,
                        "totalCount": len(retrieved_results)
                    }
                })

            # Emit chunk-evaluation event
            await thinking_callback(chunk_eval_event)

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
        thinking_callback=None,
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
        
        try:
            chat_response = self.llm.chat(
                [{"role": "user", "content": reflect_prompt}], json_mode=True
            )
            response_content = chat_response.content
            tokens_used = chat_response.total_tokens
        except Exception as e:
            if thinking_callback:
                asyncio.create_task(thinking_callback({
                    "eventType": "research-failures",
                    "subEventType": "reflection-llm-failure",
                    "error": str(e),
                    "details": {
                        "originalQuery": original_query,
                        "stage": "gap-query-generation"
                    }
                }))
            return "Failed to generate reflection due to LLM error.", [], 0

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
        except json.JSONDecodeError as e:
            # Fallback if the model returned something else
            if thinking_callback:
                asyncio.create_task(thinking_callback({
                    "eventType": "research-failures",
                    "subEventType": "json-parsing-failure",
                    "error": str(e),
                    "details": {
                        "stage": "gap-query-generation",
                        "response": response_content[:200] + "..." if len(response_content) > 200 else response_content
                    }
                }))
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
        thinking_callback=None,
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
        
        try:
            chat_response = self.llm.chat([{"role": "user", "content": prompt_text}])
            summary_text = chat_response.content
            tokens_used = chat_response.total_tokens
            return summary_text, tokens_used
        except Exception as e:
            if thinking_callback:
                asyncio.create_task(thinking_callback({
                    "eventType": "research-failures",
                    "subEventType": "summary-generation-failure",
                    "error": str(e),
                    "details": {
                        "originalQuery": original_query,
                        "chunkCount": len(all_chunks)
                    }
                }))
            return f"Failed to generate summary due to error: {str(e)}", 0

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

        try:
            # 1) Generate sub-queries
            sub_queries, used_token = self._generate_sub_queries(original_query)
            total_tokens += used_token
        except Exception as e:
            await thinking_callback({
                "eventType": "research-failures",
                "subEventType": "sub-query-generation-failure",
                "error": str(e),
                "details": {
                    "originalQuery": original_query
                }
            })
            # Use original query as fallback
            sub_queries = [original_query]

        # Let caller know which sub-questions we ended up with
        await thinking_callback(
            {
                "eventType": "questions-generated",
                "questions": sub_queries,
            }
        )

        if not sub_queries:
            # No sub-queries means we can produce a final answer right away
            # But let's say no relevant info
            await thinking_callback(
                {
                    "eventType": "final-answer",
                    "answer": f"No sub-queries needed. Possibly no relevant info for '{original_query}'",
                    "relevant_chunks": [],
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
                        session_id=kwargs.get("file_index_session_id"),
                    )
                )

            # Wait for all parallel searches
            try:
                search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            except Exception as e:
                await thinking_callback({
                    "eventType": "research-failures",
                    "subEventType": "parallel-search-failure",
                    "error": str(e),
                    "details": {
                        "iteration": iteration + 1,
                        "queries": sub_gap_queries
                    }
                })
                break

            # Merge all results
            search_res_from_vectordb = []
            for idx, result in enumerate(search_results):
                if isinstance(result, Exception):
                    await thinking_callback({
                        "eventType": "research-failures",
                        "subEventType": "individual-search-failure",
                        "error": str(result),
                        "details": {
                            "iteration": iteration + 1,
                            "query": sub_gap_queries[idx] if idx < len(sub_gap_queries) else "unknown"
                        }
                    })
                    continue
                
                res, consumed_token = result
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

            try:
                reason, new_questions, consumed_token = self._generate_gap_queries(
                    original_query, all_sub_queries, all_search_res, thinking_callback
                )
                total_tokens += consumed_token
            except Exception as e:
                await thinking_callback({
                    "eventType": "research-failures",
                    "subEventType": "gap-query-generation-failure",
                    "error": str(e),
                    "details": {
                        "iteration": iteration + 1,
                        "originalQuery": original_query
                    }
                })
                # Stop iterations on gap query failure
                break

            # If new_questions is not empty => we have another iteration
            if new_questions:
                await thinking_callback(
                    {
                        "eventType": "reflection",
                        "step": iteration + 1,
                        "reflection": reason or "Additional search needed.",
                    }
                )
                await thinking_callback(
                    {
                        "eventType": "questions-generated",
                        "questions": new_questions,
                    }
                )

                all_sub_queries.extend(new_questions)
                sub_queries = new_questions
            else:
                # No new questions => final answer scenario
                await thinking_callback(
                    {
                        "eventType": "reflection",
                        "step": iteration + 1,
                        "reflection": reason or "No further queries needed.",
                    }
                )
                break

        # If we exit the loop, let's produce a final answer event:
        # 4) Produce final summary
        try:
            summary_text, sum_tokens = self._generate_summary(
                original_query, all_sub_queries, all_search_res, thinking_callback
            )
            total_tokens += sum_tokens
        except Exception as e:
            await thinking_callback({
                "eventType": "research-failures",
                "subEventType": "final-summary-failure",
                "error": str(e),
                "details": {
                    "originalQuery": original_query,
                    "totalChunks": len(all_search_res)
                }
            })
            summary_text = f"Failed to generate final summary: {str(e)}"

        # Format relevant chunks to the requested final structure
        relevant_chunks_data = []
        for r in all_search_res:
            relevant_chunks_data.append(
                {"file_path": r.reference, "relevant_content": r.text}
            )

        await thinking_callback(
            {
                "eventType": "final-answer",
                "answer": summary_text,
                "relevant_chunks": relevant_chunks_data,
            }
        )

        additional_info = {"all_sub_queries": all_sub_queries}
        return all_search_res, total_tokens, additional_info

    def query(self, query: str, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
        """
        1) Retrieves relevant chunks for 'query'.
        2) Summarizes them and returns final answer + retrieval results.
        """
        try:
            all_retrieved_results, n_token_retrieval, additional_info = self.retrieve(
                query, **kwargs
            )
        except Exception as e:
            thinking_callback = kwargs.get("thinking_callback", lambda x: None)
            asyncio.create_task(thinking_callback({
                "eventType": "research-failures",
                "subEventType": "retrieve-orchestration-failure",
                "error": str(e),
                "details": {
                    "query": query,
                    "method": "query"
                }
            }))
            return (
                f"Failed to retrieve information due to error: {str(e)}",
                [],
                0,
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
