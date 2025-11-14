# LLMAssignment

First download vocabulary and embedding matrix through this google drive link : - https://drive.google.com/drive/folders/1kZeRQKxZv9FhIFUhpNN880g-LxfDdNVk?usp=drive_link
As i have remove that part of code

If you want to avoid training, then mention the correct CKPT_PATH , download epoch_10.pt and mention its location in CKPT_PATH
Download vocab.pkl from link , and and its path to VOCAB_SAVE_PATH
Download embedding from link , and add its path to EMBEDDING_SAVE_PATH

Then simply run the notebook main_beam_generate, which will have Training portion, generate function and beam search decoding function

For gradient accumulation , or KV caching or gradient checkpointing, you have to copy that file code and paste it in the %%writefile cell.
You can also find generate function specific to KV caching in commented portion of KV-caching
