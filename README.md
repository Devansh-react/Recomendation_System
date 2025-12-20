# Recomendation_System
A RAG LLM Based recommendation system 


User Query
   ↓
LLM (decides what to do)
   ↓
Calls RAG tool
   ↓
Tool does:
   - embed query
   - vector search
   - return structured results
   ↓
LLM formats final response
