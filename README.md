# selex-analyze-poc

A Python project managed with [uv](https://docs.astral.sh/uv/) and [Nix](https://nixos.org/) using [uv2nix](https://github.com/pyproject-nix/uv2nix).

## Getting Started

After initializing this template, run the init script to set your project name:

```bash
# Initialize with your project name
nix run github:zmblr/toolz#init-template -- my-project-name .

# Or if you cloned toolz locally
nix run .#init-template -- my-project-name .
```

## Prerequisites

- [Nix](https://nixos.org/download.html) with flakes enabled
- [direnv](https://direnv.net/) (optional, for automatic environment activation)

## Development Shells

This template provides two development shells (no `default` - explicit choice required):

| Shell    | Command                | Use Case                                                 |
| -------- | ---------------------- | -------------------------------------------------------- |
| `impure` | `nix develop .#impure` | Development, editable installs, `uv pip install` support |
| `pure`   | `nix develop .#pure`   | CI/deployment, pure Nix-managed environment              |

### Impure Shell (Development)

```bash
nix develop .#impure

# Or with direnv (recommended)
direnv allow
```

Features:

- Automatic `.venv` creation and activation
- Auto-creates/updates `uv.lock` when needed
- Supports `uv add`, `uv pip install`, `uv sync`
- Editable installs for immediate code changes

### Pure Shell (CI/Deployment)

```bash
nix develop .#pure
```

Features:

- Reproducible Nix-managed environment
- No mutable state (`.venv` not used)
- Suitable for CI pipelines and deployment testing
- Requires `uv.lock` to exist (run `impure` first)

## Quick Start

```bash
# 1. Initialize template
nix run github:zmblr/toolz#init-template -- my-project .

# 2. Enter development shell (creates uv.lock)
nix develop .#impure
# Or: direnv allow

# 3. Add dependencies
uv add requests

# 4. Install extra packages
uv pip install some-package
```

## Building the Package

```bash
# Build the default package (creates a virtual environment)
nix build

# Build with all optional dependencies
nix build .#full

# Run the package directly
nix run
```

## Formatting and Linting

```bash
# Format all code
nix fmt

# Run linters
ruff check src/
mypy src/
```

## Project Structure

```
.
├── flake.nix           # Nix flake definition
├── pyproject.toml      # Python project configuration
├── uv.lock             # uv lock file (auto-generated in impure shell)
├── nix/
│   ├── uv2nix.nix      # uv2nix workspace configuration
│   ├── packages.nix    # Package outputs
│   ├── shell.nix       # Development shell configuration
│   └── formatter.nix   # Code formatting configuration
└── src/
    └── selex_analyze_poc/
        └── __init__.py
```

## Nix Flake Outputs

| Output             | Description                                   |
| ------------------ | --------------------------------------------- |
| `packages.default` | Virtual environment with default dependencies |
| `packages.full`    | Virtual environment with all dependencies     |
| `devShells.impure` | Development shell with editable installs      |
| `devShells.pure`   | Pure Nix-managed shell (CI/deployment)        |

## How It Works

1. **uv** manages Python dependencies and generates `uv.lock`
2. **uv2nix** reads the lock file and generates Nix derivations
3. **pyproject-nix** provides build infrastructure
4. **flake-parts** organizes the flake into modular components

### Shell Comparison

| Feature               | `impure` | `pure` |
| --------------------- | -------- | ------ |
| Reproducible          | -        | +      |
| Editable installs     | +        | -      |
| `uv pip install`      | +        | -      |
| Auto venv activation  | +        | N/A    |
| Auto `uv.lock` update | +        | -      |
| CI suitable           | -        | +      |

## Updating Dependencies

```bash
# In impure shell, dependencies sync automatically
nix develop .#impure

# Update a specific package
uv lock --upgrade-package requests

# Update all packages
uv lock --upgrade
```

## License

MIT
