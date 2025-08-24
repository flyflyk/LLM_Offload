def generate_prompt(input_len):
    """Generates a prompt of a specific token length."""
    base = "Infinitely write a never-ending story for the following prompt. The salt spray was a constant companion to Thomas, the keeper of the Porthgarrow Lighthouse."
    prompt = base.split()
    multiplier = (input_len // len(prompt)) + 1
    return " ".join((prompt * multiplier)[:input_len])
