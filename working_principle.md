On startup (python main.py)                                                                                                                                                                 
                                                                                                                                                                                              
  main.py starts FastAPI                                                                                                                                                                      
    │                                                                                                                                                                                         
    ├─ asyncio.create_task(warmup_model())                                                                                                                                                    
    │    └─ Sends a dummy prompt to Ollama → loads 4GB model into GPU VRAM                                                                                                                    
    │       Takes 60–90 seconds. Runs in background.                                                                                                                                          
    │                                                                                                                                                                                         
    ├─ asyncio.create_task(warm_all_caches("Apr_2026"))    ← lib/live_data_context.py                                                                                                         
    │    └─ 3 concurrent MongoDB queries:                                                                                                                                                     
    │         _populate_globals()      → list_collection_names() on both DBs                                                                                                                  
    │         _populate_docs()         → $sample 3 docs from Apr_2026                                                                                                                         
    │         _populate_values()       → 6 concurrent $group queries (country, city, OS, browser, owner, app)                                                                                 
    │       Takes ~2–5s. Done long before the LLM is warm.                                                                                                                                    
    │                                                                                                                                                                                         
    ├─ asyncio.create_task(refresh_schema_cache("Apr_2026"))   ← lib/schema_discovery.py                                                                                                      
    │    └─ Samples 10 docs, extracts all field paths, embeds into vector store                                                                                                               
    │                                                                                                                                                                                         
    └─ asyncio.create_task(index_all_examples_async())    ← lib/query_examples.py                                                                                                             
         └─ Embeds all saved past queries into examples vector store                                                                                                                          
                                                                                                                                                                                              
  On user query (POST /api/query)                                                                                                                                                             
                                                                                                                                                                                              
  main.py: POST /api/query                                                                                                                                                                    
    │             
    ├─ 1. collection_resolver.py                                                                                                                                                              
    │      "sessions from India in Oct 2025" → detects "Oct 2025" → "Oct_2025"
    │                                         
    ├─ 2. query_generator.generate_query()                                                                                                                                                    
    │      │                                  
    │      ├─ detect_relevant_databases()                                                                                                                                                     
    │      │    "India" → needs_stream=True, needs_appconfigs=False
    │      │                                                                                                                                                                                  
    │      ├─ retrieve_schema_context()   ← schema_discovery.py + vector_store.py
    │      │    Embeds question → finds top-20 relevant field descriptions                                                                                                                    
    │      │    Returns: "clientInfo.country_name (string): Brazil ..."                                                                                                                       
    │      │                                  
    │      ├─ build_system_prompt()       ← schemas.py                                                                                                                                        
    │      │    Combines: static rules + operation type examples + schema RAG fields                                                                                                          
    │      │                                                                                                                                                                                  
    │      ├─ find_similar_examples_vector()   ← query_examples.py                                                                                                                            
    │      │    Finds 2 past queries similar to this question                                                                                                                                 
    │      │                                                                                                                                                                                  
    │      ├─ get_live_context("Oct_2025")  ← live_data_context.py  ← NEW
    │      │    Returns from in-memory cache:                                                                                                                                                 
    │      │      - Available collections list
    │      │      - 3 real stripped session documents                                                                                                                                         
    │      │      - Top 30 country names (with exact spelling)                                                                                                                                
    │      │      - OS/browser/owner/app value lists                                                                                                                                          
    │      │    Schedules background refresh if stale (no blocking)                                                                                                                           
    │      │                                  
    │      ├─ Assembles user message:                                                                                                                                                         
    │      │    [few-shot examples] + [conversation history] + [live context] + [question]                                                                                                    
    │      │                                                                                                                                                                                  
    │      └─ Retry loop (up to 3 attempts):                                                                                                                                                  
    │           → LLM generates JSON                                                                                                                                                          
    │           → _extract_json() parses it                                                                                                                                                   
    │           → _validate_structure() checks shape + sets defaults
    │           → _fix_query_obj() corrects $limit placement                                                                                                                                  
    │           → _validate_field_names() checks against known fields
    │           → If fail: _build_correction_prompt() → retry with error details                                                                                                              
    │           → If pass: return query_obj                                                                                                                                                   
    │                                                                                                                                                                                         
    ├─ 3. query_executor.execute_query()                                                                                                                                                      
    │      │                                                                                                                                                                                  
    │      ├─ Dispatch by operation:                                                                                                                                                          
    │      │    countDocuments → count_documents() — exact count, no $limit issue                                                                                                             
    │      │    find          → find().limit().sort()                                                                                                                                         
    │      │    distinct      → distinct()    
    │      │    aggregate     → aggregate() with _normalize_pipeline()                                                                                                                        
    │      │                     which expands "Bogota" → regex matching "Bogotá"                                                                                                             
    │      │                                                                                                                                                                                  
    │      └─ Returns: {results, summary, explanation, resultLabel}                                                                                                                           
    │                                                                                                                                                                                         
    ├─ 4. summarize_results()   ← result_summarizer.py                                                                                                                                        
    │      LLM analyzes results in plain English                                                                                                                                              
    │                                                                                                                                                                                         
    └─ 5. Background tasks (non-blocking):
           save_query()                 → MongoDB chat history                                                                                                                                
           add_turn()                   → session memory for follow-up questions                                                                                                              
           save_successful_query()      → RAG example store (improves future queries)                                                                                                         
                                                                                                                                                                                              
  ---                                                                                                                                                                                         
  Detailed architecture                                                                                                                                                                       
                                                                                                                                                                                              
  ┌─────────────────────────────────────────────────────────────────┐                                                                                                                         
  │                          main.py (FastAPI)                      │                                                                                                                         
  │  POST /api/query   POST /api/analyze   GET /api/health          │
  └────────────────────────────┬────────────────────────────────────┘
                               │                                                                                                                                                              
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼                                                                                                                                           
     query_generator    query_executor      result_summarizer
     (NL → JSON)        (JSON → results)    (results → prose)                                                                                                                                 
            │                  │          
      ┌─────┴──────┐           ├── countDocuments()  exact count                                                                                                                              
      │            │           ├── find()            filtered docs                                                                                                                            
      │      schemas.py        ├── distinct()        unique values
      │      (rules +          └── aggregate()       pipelines                                                                                                                                
      │       examples)              │                                                                                                                                                        
      │                        _normalize_pipeline()
      │                        └── diacritic expansion                                                                                                                                        
      │                            loggedInUserData strip                                                                                                                                     
      │                                                                                                                                                                                       
      ├── live_data_context.py   ← NEW — ground truth from live DB                                                                                                                            
      │     ┌─────────────────────────────────────────────────┐                                                                                                                               
      │     │  In-memory cache (per collection)               │                                                                                                                               
      │     │  documents:  3 real stripped session docs        │                                                                                                                              
      │     │  countries:  top-30 by frequency                │                                                                                                                               
      │     │  os_names:   all distinct                       │
      │     │  browsers:   all distinct                       │                                                                                                                               
      │     │  owners:     top-30 active                      │
      │     │  app_names:  top-30 by frequency                │                                                                                                                               
      │     │  TTL: docs=30min, values=60min                  │                                                                                                                               
      │     │  Refresh: background task, stampede-safe        │                                                                                                                               
      │     └─────────────────────────────────────────────────┘                                                                                                                               
      │                                                                                                                                                                                       
      ├── schema_discovery.py   field paths from live docs                                                                                                                                    
      │     └── vector_store.py (schema)   semantic field search                                                                                                                              
      │                                                                                                                                                                                       
      ├── query_examples.py     past successful queries                                                                                                                                       
      │     └── vector_store.py (examples)  semantic example search                                                                                                                           
      │                                                                                                                                                                                       
      └── collection_resolver.py  "October 2025" → "Oct_2025"                                                                                                                                 
                                                                                                                                                                                              
  MongoDB:                                
    stream-datastore   — monthly collections (Apr_2026, Mar_2026, …)
    appConfigs         — one collection per owner username                                                                                                                                    
                                              
  Token budget per query (8192 context window)                                                                                                                                                
                                                                                                                                                                                              
  ┌────────────────────────────────────────────────┬──────────────────────────┐                                                                                                               
  │                   Component                    │          Tokens          │                                                                                                               
  ├────────────────────────────────────────────────┼──────────────────────────┤                                                                                                               
  │ System prompt (rules + operation examples)     │ ~900                     │                                                                                                               
  ├────────────────────────────────────────────────┼──────────────────────────┤
  │ Schema RAG (top-20 relevant fields)            │ ~400                     │
  ├────────────────────────────────────────────────┼──────────────────────────┤
  │ Few-shot examples (2 past queries)             │ ~400                     │
  ├────────────────────────────────────────────────┼──────────────────────────┤                                                                                                               
  │ Live context (NEW) — doc samples + value lists │ ~350                     │
  ├────────────────────────────────────────────────┼──────────────────────────┤                                                                                                               
  │ User message (question + timestamps)           │ ~80                      │                                                                                                               
  ├────────────────────────────────────────────────┼──────────────────────────┤
  │ Response budget                                │ ~800                     │                                                                                                               
  ├────────────────────────────────────────────────┼──────────────────────────┤
  │ Total                                          │ ~2930 — well within 8192 │                                                                                                               
  └────────────────────────────────────────────────┴──────────────────────────┘
