def generate_prompt(input_len):
    """Generates a prompt of a specific token length."""
    base_prompt = "Infinitely write a never-ending story for the following prompt. The salt spray was a constant companion to Thomas, the keeper of the Porthgarrow Lighthouse."
    prompt_words = base_prompt.split()
    multiplier = (input_len // len(prompt_words)) + 1
    return " ".join((prompt_words * multiplier)[:input_len])
