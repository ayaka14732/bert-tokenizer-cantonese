from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('vocab-bart-base-cantonese.txt')
tokenizer.push_to_hub('Ayaka/bart-base-cantonese')
