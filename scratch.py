def rewrite_dual(query_obj, results1, results2):
    q1, q2 = query_obj["queries"]
    
    # We don't guarantee q1 is stream and q2 is appConfigs, but usually it is.
    merged = None
    if merge_key := query_obj.get("mergeKey"):
        # Find which is stream and which is config
        r_stream = results1
        r_config = results2
        q_config = q2
        if q1.get("database") == "appConfigs":
            r_stream = results2
            r_config = results1
            q_config = q1
            
        owner_from_config = str(q_config.get("collection", ""))
        
        combined_config = {}
        for doc in r_config:
            combined_config.update(doc)
            
        config_map = {owner_from_config: combined_config} if owner_from_config else {}
        
        # Sometimes LLM outputs owner in _id if it grouped
        for doc in r_config:
            doc_owner = str(doc.get(merge_key) or doc.get("_id", ""))
            if doc_owner and doc_owner not in ("usersinfo", "default", "InfoToConstructUrls"):
                config_map[doc_owner] = doc
                
        merged = []
        for doc in r_stream:
            app_info = doc.get("appInfo") or {}
            owner    = str(doc.get(merge_key) or app_info.get("owner", "") or doc.get("_id", "") or "")
            enriched = dict(doc)
            if owner and owner in config_map:
                enriched["_configData"] = config_map[owner]
            merged.append(enriched)
    return merged
