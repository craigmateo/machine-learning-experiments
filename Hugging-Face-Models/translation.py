"""
This script uses a Hugging Face Transformer model (NLLB) to translate text
from one language to another.

Step-by-step:

1. Suppress Logging Noise
   - Reduces unnecessary warnings from the transformers library.

2. Import Required Libraries
   - pipeline: High-level API for running pretrained models easily.
   - torch: Used here to specify the model's tensor data type.

3. Initialize Translation Pipeline
   - task="translation": Specifies that this pipeline performs translation.
   - model="facebook/nllb-200-distilled-600M":
       * A multilingual model trained to translate between 200+ languages.
       * "distilled" means it is a smaller, faster version of a larger model.
   - torch_dtype=torch.bfloat16:
       * Uses a lower-precision format to reduce memory usage.
       * Works best on supported hardware (modern CPUs/GPUs).

   NOTE:
   - If loading locally ("./models/..."), the directory must contain all
     required model files (config, weights, tokenizer, etc.).
   - Otherwise, use the Hugging Face model name directly.

4. Define Input Text
   - A multi-line string containing English sentences to translate.

5. Perform Translation
   - translator(...) runs the model on the input text.
   - src_lang="eng_Latn": Source language is English (Latin script).
   - tgt_lang="fra_Latn": Target language is French (Latin script).

6. Output Format
   - The result is a list of dictionaries:
       [{"translation_text": "..."}]

7. Print Result
   - Displays the translated text in the console.

Key Concepts:

- Pipeline Abstraction:
  Simplifies model usage by handling tokenization, inference, and decoding.

- Multilingual Translation (NLLB):
  The model supports many languages using standardized language codes
  like "eng_Latn", "fra_Latn", etc.

- Language Codes:
  Format is generally:
      <language>_<script>
  Example:
      eng_Latn = English (Latin alphabet)
      fra_Latn = French (Latin alphabet)

Notes / Potential Issues:

- Local Model Path:
  "./models/facebook/nllb-200-distilled-600M" may cause errors unless the
  model is properly downloaded and structured.

- torch_dtype Compatibility:
  bfloat16 may not be supported on all machines. If errors occur, remove it.

- Large Model:
  This model (~600M parameters) may take time to download and load.

This script demonstrates a basic but powerful use case:
→ multilingual machine translation using a pretrained transformer model.
"""

from transformers.utils import logging
logging.set_verbosity_error()

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = """My puppy is adorable,
Your kitten is cute.
Her panda is friendly.
His llama is thoughtful.
We all have nice pets!"""

inputs = tokenizer(text, return_tensors="pt")

translated_tokens = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.convert_tokens_to_ids("fra_Latn"),
    max_length=200
)

translated_text = tokenizer.batch_decode(
    translated_tokens,
    skip_special_tokens=True
)[0]

print(translated_text)