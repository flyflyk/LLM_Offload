def generate_prompt(input_len, tokenizer) -> str:
    base = "Infinitely write a never-ending story for the following prompt. The salt spray was a constant companion to Thomas, the keeper of the Porthgarrow Lighthouse."
    base_tokens = tokenizer.encode(base)
    
    multiplier = (input_len // len(base_tokens)) + 1
    token_ids = (base_tokens * multiplier)[:input_len]
    
    return tokenizer.decode(token_ids, skip_special_tokens=True)
