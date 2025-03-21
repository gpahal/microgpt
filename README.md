# microgpt

A micro GPT implementation and training pipeline in PyTorch for learning purposes.

```python
from microgpt.model import (
    load_model,
    PretrainedGPT2ModelConfig,
    PretrainedGPT2ModelType,
)

model = await load_model(
    config=PretrainedGPT2ModelConfig(
        model_type=PretrainedGPT2ModelType.GPT_2,
    ),
)
generated_text = model.generate_text(
    text="Hi, I'm a language model,",
    max_new_tokens=50,
)
```

## Features

- [x] Tokenizer
  - [x] Loading pretrained gpt tokenizers
  - [x] Training custom byte-pair encoding tokenizers
  - [x] Loading custom byte-pair encoding tokenizers from files
- [x] Micro GPT model implementation
  - [x] Loading pretrained gpt models
  - [x] Training custom gpt models with support for DDP
  - [x] Training checkpoints
  - [x] Loading custom gpt models from files
- [x] Training data loading using text, files, urls or huggingface datasets
- [ ] Reproducing GPT-2 with a custom tokenizer and model
- [ ] Finetuning

## Usage

- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

- Install [make](https://www.gnu.org/software/make/)

- Setup a virtual environment

```sh
uv venv --python 3.12
source .venv/bin/activate
```

- Install dependencies

```sh
make sync
```

- Go through the [notebooks](notebooks) to understand how to use the library.

## Acknowledgements

- [Neural Networks: Zero to Hero by Andrej Karpathy](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [karpathy/minbpe](https://github.com/karpathy/minbpe)
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
