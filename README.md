# BERT Tokenizer Cantonese

## Motivation

The Chinese BERT tokenizer is widely used among many Chinese NLP models, including the [Chinese BART model](https://huggingface.co/fnlp/bart-base-chinese). However, it cannot be directly applied to Cantonese because it is mainly designed to tokenise Simplified Chinese, whereas Hong Kong Cantonese is mainly written in Traditional Chinese. Moreover, it lacks those Chinese characters that are normally used only in Cantonese, such as '冧', '嚿' and '曱'. Therefore, this project provides a BERT Tokenizer with vocabulary tailored for Cantonese.

## Build

```sh
sudo apt install -y git-lfs
python3.10 -m venv ./venv
. ./venv/bin/activate
pip install -U pip
pip install -U wheel
pip install "jax[tpu]==0.3.23" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -r requirements.txt
huggingface-cli login
```

```sh
wget https://raw.githubusercontent.com/StarCC0/dict/4cf962546a1faf680ac8b15e8ff5d501c98a96ea/STCharacters.txt
wget https://raw.githubusercontent.com/StarCC0/dict/4cf962546a1faf680ac8b15e8ff5d501c98a96ea/TSCharacters.txt
gdown 1F2XlacTo3dTyNItsWY65ocJFgyA9u5CW  # lihkg-1-2850000-processed-dedup.csv.xz
xz --decompress lihkg-1-2850000-processed-dedup.csv.xz
# Manually download `yue.txt` from the Cantonese translation dataset
```

```sh
python build_vocab_mapping.py
python add_cantonese_tokens.py
python replace_embedding.py
upload_to_hub.py
# Manually upload `embed_params.dat` to `Ayaka/bart-base-cantonese`
```
