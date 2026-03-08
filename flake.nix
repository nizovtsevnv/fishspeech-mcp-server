{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };

        cargoToml = builtins.fromTOML (builtins.readFile ./Cargo.toml);
        pname = cargoToml.package.name;
        version = cargoToml.package.version;

        cargoHash = "sha256-JNB3KI0USNKGCYfXzR1w4izsiQ0IoqwtMKG+5zolI1U=";

        rustToolchain = pkgs.rust-bin.stable.latest.default;
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            rustToolchain
            pkgs.cmake
            pkgs.pkg-config
            pkgs.libclang
            pkgs.llvmPackages.libclang
            pkgs.libopus
          ];

          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

          shellHook = ''
            # Set up git hooks if .git exists
            if [ -d .git ]; then
              mkdir -p .git/hooks
              cat > .git/hooks/pre-commit << 'HOOK'
#!/usr/bin/env bash
set -e
cargo fmt -- --check
cargo clippy -- -D warnings
cargo test
HOOK
              chmod +x .git/hooks/pre-commit
            fi
          '';
        };

        packages = {
          default = pkgs.rustPlatform.buildRustPackage {
            inherit pname version;
            src = ./.;
            inherit cargoHash;

            nativeBuildInputs = with pkgs; [
              cmake
              pkg-config
              llvmPackages.libclang
            ];
            buildInputs = [ pkgs.libopus ];

            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

            meta = with pkgs.lib; {
              description = "Text-to-speech MCP server powered by Fish Speech 1.5";
              license = licenses.mit;
            };
          };
        };
      }
    );
}
