{
  perSystem = {
    workspace,
    pythonSet,
    ...
  }: {
    packages = {
      # Default package: virtual environment with default dependencies only
      default = pythonSet.mkVirtualEnv "selex-analyze-poc-env" workspace.deps.default;

      # Full package: virtual environment with all dependencies (including optional)
      full = pythonSet.mkVirtualEnv "selex-analyze-poc-full-env" workspace.deps.all;
    };
  };
}
