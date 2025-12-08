{inputs, ...}: {
  perSystem = {
    pkgs,
    lib,
    ...
  }: let
    # Load the uv workspace from the project root
    workspace = inputs.uv2nix.lib.workspace.loadWorkspace {
      workspaceRoot = ./..;
    };

    # Create an overlay from the workspace for regular (non-editable) builds
    overlay = workspace.mkPyprojectOverlay {
      sourcePreference = "wheel";
    };

    # Create an editable overlay for development
    # Uses $REPO_ROOT environment variable set in shell.nix
    editableOverlay = workspace.mkEditablePyprojectOverlay {
      root = "$REPO_ROOT";
    };

    # Select Python version
    python = pkgs.python312;

    # Create the base Python package set using pyproject-nix
    baseSet = pkgs.callPackage inputs.pyproject-nix.build.packages {
      inherit python;
    };

    # Apply overlays to create the final Python set for production builds
    # Use overlays.wheel for faster builds (prebuilt wheels from PyPI)
    # Use overlays.sdist if you need to build everything from source
    pythonSet = baseSet.overrideScope (
      lib.composeManyExtensions [
        inputs.pyproject-build-systems.overlays.wheel
        overlay
      ]
    );

    # Apply editable overlay for development builds
    pythonSetEditable = pythonSet.overrideScope editableOverlay;
  in {
    _module.args = {
      inherit workspace pythonSet pythonSetEditable python;
    };
  };
}
