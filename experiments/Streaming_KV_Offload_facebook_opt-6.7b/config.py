CHOSEN_MODEL = "facebook/opt-6.7b"
MAX_TOKENS = 20
PROMPT_LIST = [
    (
    "Infinitely write a never-ending story for the following prompt. "
    "The salt spray was a constant companion to Thomas, the keeper of the Porthgarrow Lighthouse. "
    "For thirty years, its beam had sliced through the darkest nights, a"
    ),
    "Translate the following English text to French: 'Hello, how are you today?'",
    "Write a short poem about a cat watching the rain.",
    "What is the capital of Japan?",
]
BATCH_SIZE = 4
ENABLE_STREAMING = True
ENABLE_KV_OFFLOAD = True
PROMPT_LOG = False
OFFLOAD_FOLDER = "offload_dir"
MAX_CPU_OFFLOAD_RAM_GB = 8
DEVICE = "cuda"
LOG_FILE = "log.log"
