import json
from typing import Tuple, List, Dict, Any
from app.services.llm import generate_answer
from app.config.settings import prompts
from app.core.logger import RequestLogger

class CitationEnforcer:
    """
    PRODUCTION CRITICAL:
    Ensures LLM answers are grounded in retrieved evidence.
    Refuses to generate answers when context doesn't support them.
    """
    
    def __init__(self, logger: RequestLogger):
        self.logger = logger
    
    def check_citations(
        self,
        question: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        threshold: float = 0.7
    ) -> Tuple[bool, float, str]:
        """
        Verify if generated answer is actually supported by retrieved context.
        
        Args:
            question: User's question
            answer: LLM-generated answer
            retrieved_chunks: List of retrieved documents
            threshold: Confidence threshold (0.0-1.0)
        
        Returns:
            (is_supported: bool, confidence: float, reasoning: str)
        """
        
        # If no context retrieved, answer cannot be supported
        if not retrieved_chunks:
            return False, 0.0, "No context retrieved to support answer"
        
        # Build context string from chunks
        context = "\n---\n".join([
            f"[{c['source']}]\n{c['text']}"
            for c in retrieved_chunks[:3]  # Use top 3 chunks
        ])
        
        # Use LLM to judge if answer is supported
        check_prompt = prompts["citation_check_prompt"].format(
            question=question,
            context=context,
            answer=answer
        )
        
        try:
            judgment = generate_answer(check_prompt)
            
            # Parse LLM's JSON judgment
            result = json.loads(judgment)
            
            is_supported = result.get("is_supported", False)
            confidence = float(result.get("confidence", 0.0))
            reasoning = result.get("reasoning", "")
            
            # Log the judgment
            self.logger.log_quality_check(
                is_citation_supported=is_supported,
                confidence=confidence,
                reasoning=reasoning
            )
            
            return is_supported, confidence, reasoning
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # If citation check fails, assume answer is unsupported (fail-safe)
            self.logger.log_error(
                error_type="citation_check_failed",
                error_msg=f"Failed to parse citation judgment: {str(e)}",
                traceback=str(e)
            )
            return False, 0.0, "Could not verify citations"
    
    def enforce_citations(
        self,
        question: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        best_score: float,
        threshold: float = 0.7
    ) -> Tuple[str, bool]:
        """
        PRODUCTION PATTERN:
        If answer is not supported by context, refuse to generate it.
        
        Args:
            question: User's question
            answer: LLM-generated answer
            retrieved_chunks: Retrieved documents
            best_score: Best retrieval score (quality of context)
            threshold: Confidence threshold
        
        Returns:
            (final_answer: str, is_hallucination: bool)
        """
        
        # If retrieval quality is very low, don't trust the answer
        if best_score < 0.3:
            fallback = (
                "I don't have reliable information about this in my knowledge base. "
                "Please consult with a local agricultural expert or extension service."
            )
            return fallback, True
        
        # Check if answer is actually supported by context
        is_supported, confidence, reasoning = self.check_citations(
            question=question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            threshold=threshold
        )
        
        if not is_supported:
            fallback = (
                "I found some relevant documents, but I'm not confident enough "
                "to answer this without more reliable information. "
                "I recommend checking official agricultural resources or consulting an expert."
            )
            return fallback, True
        
        if confidence < threshold:
            # Return answer but add confidence disclaimer
            augmented_answer = f"""
{answer}

⚠️ Note: I have moderate confidence in this answer. 
For critical decisions, please verify with your agricultural extension service.
"""
            return augmented_answer, False
        
        # High confidence - answer is well-supported
        return answer, False
    
    def format_answer_with_citations(
        self,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Format answer with visible citations and sources.
        
        Example output:
        ```
        Answer: Nitrogen deficiency typically manifests as yellowing of lower leaves...
        
        Sources:
        - Agronomy-Manual.pdf (Page 42)
        - Fertilizers-Guide.pdf (Page 18)
        ```
        """
        
        if not retrieved_chunks:
            return answer
        
        # Extract unique sources
        sources = {}
        for chunk in retrieved_chunks[:3]:
            source = chunk.get("source", "Unknown")
            if source not in sources:
                sources[source] = []
            sources[source].append(chunk.get("chunk_id"))
        
        # Format with citations
        citation_section = "\n\n**Sources Used:**\n"
        for i, (source, chunk_ids) in enumerate(sources.items(), 1):
            citation_section += f"{i}. {source} (chunks: {', '.join(chunk_ids)})\n"
        
        return f"{answer}\n{citation_section}"