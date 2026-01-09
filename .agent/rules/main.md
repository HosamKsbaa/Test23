---
trigger: always_on
---

Step 1: The Data Processing Instruction (The M2 Parser)
Prompt for Copilot:

"I have uploaded the QALB dataset files: QALB-2015-L2-Dev.m2 and QALB-2015-L2-Dev.sent. Write a Python script to create a parallel training dataset. The script must: 1. Parse the M2 Format: Read lines starting with 'S' as the source (incorrect) and lines starting with 'A' as the specific edits. 2. Apply All Edits: For every 'S' line, apply every corresponding 'A' edit (spelling, grammar, syntax, punctuation) to reconstruct the fully corrected sentence. 3. Critical Reversal Logic: Apply edits in reverse order of their token offsets (from the end of the sentence to the beginning) to prevent index shifting during reconstruction. 4. Output Format: Save the results to a CSV named qalb_full_gec.csv with two columns: incorrect and correct. 5. Encoding: Use utf-8 to ensure Arabic characters, Hamzas, and Tashkeel are preserved correctly."
+2

Step 2: The Model Training Instruction (AraT5 Fine-tuning)
Prompt for Copilot:

"Using the transformers and datasets libraries, write a script to fine-tune aubmindlab/arat5-v2-base for full Arabic error correction.

Data Loading: Load the qalb_full_gec.csv generated in the previous step.

Task Trigger: Prepend the prefix 'gec_arabic: ' to every input sentence to trigger the correction mode.

Preprocessing: Use the AraT5 tokenizer with max_length=128. Use pyarabic to ensure consistent normalization of Alif and Hamza styles during tokenization.

Training Config: Use Seq2SeqTrainer with the following:

learning_rate=3e-5

num_train_epochs=5

predict_with_generate=True

evaluation_strategy='steps'

Metric: Include a function to compute GLEU (the standard for GEC) or CER (Character Error Rate) to monitor grammatical precision."

Step 3: The Manual Testing/Inference Instruction
Prompt for Copilot:

"Create a Python inference script to test the fine-tuned model on new Arabic sentences.

User Input: Accept a raw, erroneous Arabic sentence (e.g., 'ذهب الولد الى مدرسة').

Generation Params: Use model.generate with:

num_beams=5 (for high-quality grammatical search)

early_stopping=True

no_repeat_ngram_size=2

Validation: The model should output a fully corrected sentence including correct Hamzas, Ta Marbutas, and proper case endings (Nahw)."