#WIP

# Labels for clause types
SUB_DEPS = {"advcl", "ccomp", "acl", "relcl", "xcomp"}  # subordinate
CC_DEP   = "cc"                                         # coordinating conj
MARK_DEP = "mark"                                       # subordinating conj marker
# Remove incorrect CMP_DEP definition (ccomp is already in SUB_DEPS)
# Add proper comparative conjunctions
CMP_DEPS = {"prep", "pcomp", "pobj"}  # comparative elements

# Subordinating conjunctions dictionary
SUBORDINATING_CONJUNCTIONS = {
    "temporal": {
        "after", "as", "as soon as", "before", "by the time", "once",
        "since", "till", "until", "when", "whenever", "while", "now that"
    },
    "causal": {
        "as", "because", "since", "insofar as", "seeing that", "for", "now that"
    },
    "conditional": {
        "if", "even if", "in case", "provided that", "unless", "only if",
        "assuming that", "on condition that", "as long as"
    },
    "concessive": {
        "although", "even though", "though", "whereas", "while"
    },
    "purpose": {
        "so that", "in order that", "lest", "for the purpose that"
    },
    "result/consequence": {
        "so that", "so", "such that", "in such a way that"
    },
    "comparison": {
        "than", "as...as", "as if", "as though"
    },
    "manner": {
        "as", "as if", "as though", "the way", "just as"
    },
    "relative (nominal)": {
        "that", "which", "who", "whom", "whose", "what", "whatever", "whichever", "whoever"
    },
    "exception": {
        "except that", "save that"
    }
}

# Function to identify the type of a subordinating conjunction
def identify_conjunction_type(conjunction):
    conjunction = conjunction.lower()
    for conj_type, conj_set in SUBORDINATING_CONJUNCTIONS.items():
        if conjunction in conj_set:
            return conj_type
    return "unknown"

def get_detailed_tense(tok):
    """Extract detailed tense, aspect, and modality information from a verb."""
    # Get basic morphological features
    tense = tok.morph.get("Tense")
    tense = tense[0] if tense else None
    
    # Check children for auxiliaries to determine more precise tense and aspect
    aux_tokens = [child for child in tok.children if child.dep_ in ["aux", "auxpass"]]
    
    # For raining-type verbs, check if head is an auxiliary verb
    if tok.dep_ == "advcl" and tok.head.pos_ == "AUX":
        # Include the head aux verb when determining tense
        aux_tokens.append(tok.head)
    
    # Print auxiliaries for debugging
    print(f"Verb: {tok.text}, Auxiliaries: {[aux.text for aux in aux_tokens]}")
    
    # Determine tense based on auxiliaries and morphology
    if any(aux.lemma_ == "be" and aux.morph.get("Tense")[0] == "Past" for aux in aux_tokens if aux.morph.get("Tense")):
        if tok.tag_ == "VBG":
            return "Past-Prog"  # e.g., "was walking"
        elif tok.tag_ == "VBN":
            return "Past-Pass"  # e.g., "was taken" 
    
    if any(aux.lemma_ == "have" and aux.morph.get("Tense")[0] == "Past" for aux in aux_tokens if aux.morph.get("Tense")):
        if tok.tag_ == "VBN":
            return "Past-Perf"  # e.g., "had walked"
    
    if any(aux.lemma_ == "will" for aux in aux_tokens):
        return "Future"  # e.g., "will walk"
    
    if any(aux.lemma_ == "would" for aux in aux_tokens):
        return "Cond"  # e.g., "would walk" 
    
    if any(aux.lemma_ == "be" and aux.morph.get("Tense")[0] == "Pres" for aux in aux_tokens if aux.morph.get("Tense")):
        if tok.tag_ == "VBG":
            return "Pres-Prog"  # e.g., "is walking"
    
    if any(aux.lemma_ == "have" and aux.morph.get("Tense")[0] == "Pres" for aux in aux_tokens if aux.morph.get("Tense")):
        if tok.tag_ == "VBN":
            return "Pres-Perf"  # e.g., "has walked"
    
    # Default based on morphological tense
    if tense == "Pres":
        return "Simple-Pres"  # e.g., "walks"
    elif tense == "Past":
        return "Simple-Past"  # e.g., "walked"
    
    # Check for any conflicting tense markers
    if "was" in [aux.text.lower() for aux in aux_tokens] and tok.tag_ == "VBG":
        return "Past-Prog"  # Explicit check for "was + gerund"
    
    # If nothing else matches
    return tense if tense else "Unknown"

def contains_verb(subtree):
    """Check if a subtree contains any verb"""
    return any(token.pos_ == "VERB" for token in subtree)

def build_span(root, kind):
    """
    For 'indep', include everything in subtree except:
      - punctuation
      - conjoined clause after a coordinating conjunction like 'but'
      - comparative elements (like "than" phrases) that contain verbs
      - tokens under SUB_DEPS branches (except their 'mark')
    For 'sub', include entire subordinate subtree but drop its 'mark' and punctuation and cc-subtrees.
    """
    exclude = set()
    
    print(f"\nBUILDING SPAN FOR: {root.text} ({kind})")
    
    if kind == "indep":
        # Find conjoined verbs and exclude them and their subtrees
        for token in root.children:
            if token.dep_ == "conj":
                print(f"Found conjoined verb: {token.text}")
                # Find the conjunction connecting to this conj
                cc_token = None
                for t in doc:
                    if t.dep_ == CC_DEP and t.head == root and t.i < token.i:
                        cc_token = t
                        break
                
                # If we found the conjunction, exclude it and the conj verb's subtree
                if cc_token:
                    print(f"Found CC: {cc_token.text}")
                    exclude.add(cc_token)
                    for t in token.subtree:
                        print(f"Excluding from subtree: {t.text} ({t.dep_})")
                    exclude.update(token.subtree)
    
    # Handle comparative elements and prepositions
    for child in root.children:
        # Only exclude prep/pobj if they contain a verb or the word "than"
        if child.dep_ in CMP_DEPS:
            if contains_verb(child.subtree) or any(tok.text.lower() == "than" for tok in child.subtree):
                print(f"Excluding comparative with verb: {child.text}")
                exclude.update(child.subtree)
            else:
                print(f"Keeping preposition without verb: {child.text}")

    # Collect tokens
    all_subtree = list(root.subtree)
    print(f"All subtree tokens: {[t.text for t in all_subtree]}")
    
    tokens = []
    for t in all_subtree:
        if t.is_punct:
            print(f"Excluding punct: {t.text}")
            continue
        if t in exclude:
            print(f"Excluding: {t.text}")
            continue
        
        if kind == "indep":
            # find which direct child of root this token descends from
            # via ancestors
            direct = None
            for anc in t.ancestors:
                if anc.head == root:
                    direct = anc
                    break
            # if that direct child is a subordinate clause, skip all its tokens except the 'mark'
            if direct and direct.dep_ in SUB_DEPS and t.dep_ != "mark":
                print(f"Excluding subordinate: {t.text}")
                continue
        elif kind == "sub":
            if t.dep_ == "mark":
                print(f"Excluding mark: {t.text}")
                continue
        
        print(f"Including: {t.text}")
        tokens.append(t)

    if not tokens:
        print(f"No tokens found, using root only: {root.text}")
        tokens = [root]

    tokens = sorted(tokens, key=lambda x: x.i)
    span = doc[tokens[0].i : tokens[-1].i + 1]
    print(f"Final span: {span.text}")
    return span

def annotate_indep(span):
    text = span.text
    base = span.start_char
    # find subordinate spans inside this span
    inside = [
        (sp, sid) for sp, sid in sub_lookup.items()
        if sp.start_char >= base and sp.end_char <= span.end_char
    ]
    # replace in reverse order to keep indexes valid
    for sp, sid in sorted(inside, key=lambda x: x[0].start_char, reverse=True):
        s = sp.start_char - base
        e = sp.end_char   - base
        text = text[:s] + f"<EVENT {sid}>" + text[e:]
    return text.strip()

if __name__ == "__main__":

    text = "She walks to school, but he had stayed home because it was raining."
    text2 = "On the other hand, he rather liked the rain."
    # Add import for spacy if not already present
    import spacy
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    print("Model loaded successfully.")
    doc = nlp(text)

    # For debugging: print the dependency structure
    print("DEPENDENCY STRUCTURE:")
    for token in doc:
        print(f"{token.i:<3} {token.text:<10} {token.dep_:<10} {token.head.text:<10} {[child.text for child in token.children]}")

    # Collect independent-clause verbs and subordinate-clause verbs
    indep = [t for t in doc if t.pos_ == "VERB" and t.dep_ in {"ROOT","conj"} and t.head.dep_ not in SUB_DEPS]
    subord = []
    for root in indep:
        for child in root.children:
            if child.pos_ == "VERB" and child.dep_ in SUB_DEPS:
                subord.append(child)

    # Print debug for verbs
    print("\nVERBS:")
    print("Independent verbs:", [t.text for t in indep])
    print("Subordinate verbs:", [t.text for t in subord])

    # Assign IDs
    events = indep + subord
    event_id = {tok: i+1 for i, tok in enumerate(events)}

    # Build event records
    records = []
    for tok in events:
        kind = "indep" if tok in indep else "sub"
        span = build_span(tok, kind)
        records.append({
            "tok": tok,
            "span": span,
            "kind": kind,
            "tense": get_detailed_tense(tok),
            "id": event_id[tok]
        })

    # Sort by occurrence
    records.sort(key=lambda r: r["span"].start_char)

    # Store record info for easy lookup by ID
    record_by_id = {rec["id"]: rec for rec in records}

    # Build subordinate lookup for replacements
    sub_lookup = {r["span"]: r["id"] for r in records if r["kind"] == "sub"}

    # Find all conjunction markers
    conjunction_markers = {}
    for token in doc:
        if token.dep_ == MARK_DEP:
            # Find the verb this mark is associated with
            head_verb = token.head
            while head_verb.pos_ != "VERB" and head_verb.i < len(doc) - 1:
                # If head isn't a verb, try to find the actual verb
                for child in head_verb.children:
                    if child.pos_ == "VERB":
                        head_verb = child
                        break
                else:
                    # No verb found among children, move up
                    if head_verb.head != head_verb:  # Avoid infinite loop at root
                        head_verb = head_verb.head
                    else:
                        break
            
            # Find which event this verb belongs to
            for i, rec in enumerate(records):
                if rec["tok"] == head_verb:
                    conj_type = identify_conjunction_type(token.text)
                    conjunction_markers[i] = {"text": token.text, "type": conj_type}
                    break

    # Output the final results with conjunction information
    print("\nFINAL RESULTS:")
    print(f"{'ID':<3} {'Event Phrase':<45} {'Tense':<12} {'Conjunction':<10} {'Type':<15}")
    print("-"*90)
    for i, rec in enumerate(records):
        phrase = annotate_indep(rec["span"]) if rec["kind"] == "indep" else rec["span"].text
        
        # Get conjunction information if available
        conj_text = ""
        conj_type = ""
        if i in conjunction_markers:
            conj_text = conjunction_markers[i]["text"]
            conj_type = conjunction_markers[i]["type"]
        
        print(f"{rec['id']:<3} {phrase:<45} {rec['tense']:<12} {conj_text:<10} {conj_type:<15}")

    # Find linking conjunctions between events
    conjunctions = {}
    for token in doc:
        if token.dep_ == CC_DEP:
            # Find which events this conjunction connects
            head_id = None
            for rec in records:
                if token.head == rec["tok"]:
                    head_id = rec["id"]
                    break
            
            # Find the target event (the one after the conjunction)
            target_id = None
            for rec in records:
                if rec["tok"].dep_ == "conj" and rec["tok"].head == token.head and rec["tok"].i > token.i:
                    target_id = rec["id"]
                    break
            
            if head_id and target_id:
                conjunctions[(head_id, target_id)] = token.text

    # Generate the dynamic output string with conjunction types
    print("\nDYNAMIC EVENT STRING WITH CONJUNCTION TYPES:")
    # Start with the first event
    dynamic_string = f"<EVENT {records[0]['id']}>"

    # Find connections between events
    for i in range(len(records)-1):
        current_id = records[i]["id"]
        next_id = records[i+1]["id"]
        
        # Check if there's a direct conjunction between these events
        if (current_id, next_id) in conjunctions:
            conj = conjunctions[(current_id, next_id)]
            conj_type = "coordinating"  # All CC_DEP are coordinating conjunctions
            dynamic_string += f" {conj}[{conj_type}] <EVENT {next_id}>"
        # Check for embedding/subordination
        elif next_id in sub_lookup.values():
            # Find which event contains this one and get the conjunction
            for j, rec in enumerate(records):
                if rec["id"] == next_id and j in conjunction_markers:
                    conj = conjunction_markers[j]["text"]
                    conj_type = conjunction_markers[j]["type"]
                    dynamic_string += f"({conj}[{conj_type}] <EVENT {next_id}>)"
                    break
            else:
                # If no conjunction found
                dynamic_string += f"(<EVENT {next_id}>)"
        else:
            # Default connection
            dynamic_string += f" / <EVENT {next_id}>"

    print(dynamic_string)