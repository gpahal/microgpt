# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

microgpt is a micro GPT implementation and training pipeline in PyTorch. It provides a complete framework for training custom GPT-like models with distributed training support, custom tokenizers, and flexible data loading.

## Development Commands

This project uses `uv` (Astral's fast Python package manager) and provides a Makefile:

```bash
# Install all dependencies (including dev dependencies)
make sync

# Run linting checks (ruff + mypy)
make lint

# Run linting with auto-fix
make lint-fix

# Run tests
make test

# Start Jupyter Lab
make jupyter
```

## Architecture

### Core Modules

1. **Tokenizer** (`src/microgpt/tokenizer/`): Custom byte-pair encoding implementation with support for loading pretrained GPT tokenizers
2. **Model** (`src/microgpt/model/`): GPT-like transformer with RoPE support and DDP training capabilities
3. **Common** (`src/microgpt/common/`): Shared utilities including data sources, device management, and training infrastructure

### Key Design Patterns

- **Configuration-Driven**: All components use Pydantic models for type-safe configuration
- **Async I/O**: Extensive use of async/await for data loading and file operations
- **Abstract Base Classes**: Common trainer logic abstracted for reusability
- **Distributed Training**: Built-in support for multi-GPU training via DDP/torchrun

### Data Sources

The project supports multiple data source types through a unified interface:

- Text strings
- Local files
- URLs
- HuggingFace datasets

### Training Pipeline

Two-stage training approach:

1. Large-scale web data training
2. High-quality data fine-tuning with model souping

## Testing

- Framework: pytest with async support
- Coverage: pytest-cov (reports in `.coverage_html`)
- Run single test: `pytest tests/path/to/test.py::test_function_name`

## Code Quality

- **Linting**: ruff (pycodestyle, pyflakes, isort, flake8-bugbear)
- **Type Checking**: mypy in strict mode
- **Formatting**: ruff with 120-character line length
- **Docstrings**: Google style convention
