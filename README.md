Retrieval-Augmented Generation (RAG) is an AI framework that improves the accuracy and relevance of large language model (LLM) outputs by fetching, or "retrieving," trusted, up-to-date information from external data sources before generating a response. It reduces hallucinations and provides grounded, context-aware answers without needing to retrain the underlying model. How RAG Works RAG operates by enhancing LLMs with external knowledge, typically following this process: Retrieve: When a user poses a query, the system searches authorized internal documents, databases, or the web to find relevant information. Augment: The user's prompt is combined with the retrieved, relevant data to create an enhanced prompt. Generate: The LLM uses this specific context to generate an accurate, informed response.

Mean Reciprocal Rank (MRR) is a key metric for evaluating search engines or recommendation systems by measuring the average inverse rank of the first relevant document across a set of queries. It rewards systems that place the best result at the top.
The formula is:
\(\text{MRR}=\frac{1}{Q}\sum _{i=1}^{Q}\frac{1}{\text{rank}_{i}}\). 

Key Aspects of MRR:
Focus: Only cares about the first relevant document.
Range: \(0\) to \(1\). A score of \(1\) means the top result is always relevant.
Application: Ideal for systems with one, or few, "correct" answers (e.g., Q&A, voice assistants). 

Calculation Steps: 
Find the Rank (\(r_{i}\)): For each query, identify the position (1, 2, 3...) of the first relevant result.
Calculate Reciprocal Rank (RR): Calculate \(1/r_{i}\) for each query.
Average Results: Sum the RRs and divide by the total number of queries (\(Q\)). 

Example:
Query 1: First relevant result at rank 1 \(\rightarrow \) \(\frac{1}{1}=1.0\)
Query 2: First relevant result at rank 3 \(\rightarrow \) \(\frac{1}{3}\approx 0.33\)
Query 3: First relevant result at rank 2 \(\rightarrow \) \(\frac{1}{2}=0.5\)
MRR = \(\frac{1+0.33+0.5}{3}=\frac{1.83}{3}\approx \mathbf{0.61}\)
If no relevant document is found, the RR is \(0\). 

I integrated MRR (Mean Reciprocal Rank) as an evaluation mechanism to analyze retrieval
performance. Instead of only retrieving documents, I measured how effectively the system ranked
relevant information. By using MRR, I was able to evaluate and improve the quality of retrieved results
and understand how ranking impacts final answer accuracy. This step strengthened the evaluation
framework of the RAG system rather than changing the retrieval logic itself.
