{
  perSystem = {
    config,
    pkgs,
    ...
  }: let
    runpod-provider = pkgs.terraform_1.plugins.mkProvider rec {
      version = "1.0.1";
      owner = "decentralized-infrastructure";
      repo = "terraform-provider-runpod";
      rev = "v${version}";
      hash = "sha256-FdWjV2WygRRN6lIdc9iIHvdkuHcYHqywr4vzK/El2fE=";
      provider-source-address = "registry.opentofu.org/decentralized-infrastructure/runpod";
      vendorHash = "sha256-mivNrGnGhsQQZpg3kpMPVtMi5UeHYurTVEaSWAkEEPU=";
      spdx = "mit";
    };
  in {
    devShells.terraform = pkgs.mkShellNoCC {
      packages = with pkgs;
        [
          sops
          runpodctl
        ]
        ++ [config.packages.terraform];
    };
    packages = {
      terraform = pkgs.opentofu.withPlugins (p: [
        p.carlpett_sops
        runpod-provider
      ]);
    };
  };
}
