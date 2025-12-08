{
  perSystem = {
    pkgs,
    python,
    workspace,
    pythonSet,
    ...
  }: let
    # Production virtual environment (non-editable)
    virtualenv = pythonSet.mkVirtualEnv "selex-analyze-poc-env" workspace.deps.all;
  in {
    devShells = {
      impure = pkgs.mkShell {
        packages = [
          python
          pkgs.uv
        ];

        env = {
          UV_PYTHON_DOWNLOADS = "never";
        };

        shellHook = ''
          export REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)

          if [[ ! -d ".venv" ]]; then
            echo "Creating virtual environment..."
            uv venv --quiet
          fi

          source .venv/bin/activate

          if [[ ! -f "uv.lock" ]] || [[ "pyproject.toml" -nt "uv.lock" ]]; then
            echo "Updating uv.lock..."
            uv lock --quiet
          fi

          uv sync --quiet
        '';
      };

      # Pure: Nix-managed shell for CI/deployment (requires uv.lock to exist)
      pure = pkgs.mkShell {
        packages = [
          virtualenv
          pkgs.uv
        ];

        env = {
          UV_NO_SYNC = "1";
          UV_PYTHON = python.interpreter;
          UV_PYTHON_DOWNLOADS = "never";
        };

        shellHook = ''
          unset PYTHONPATH
          export REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
          export VIRTUAL_ENV="${virtualenv}"
        '';
      };
    };
  };
}
