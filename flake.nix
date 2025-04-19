{
  description = "Python3.12 dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
        let
          pkgs = import nixpkgs { inherit system; };
        in {
          devShells.default = pkgs.mkShell {
            # Set zsh as the shell
            shell = "${pkgs.zsh}/bin/zsh";

            packages = with pkgs; [
              python312Full
              python312Packages.pip
              python312Packages.virtualenv
            ];

          };
        }
    );
}

