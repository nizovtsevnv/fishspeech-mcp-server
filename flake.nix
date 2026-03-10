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

        cargoHash = "sha256-7x5l061za1kueFdR4vWXyTDZMLtriAk1wWiyyLkodnU=";

        rustToolchain = pkgs.rust-bin.stable.latest.default;

        # Static libopus for embedding into the binary
        libopusStatic = pkgs.libopus.overrideAttrs (old: {
          mesonFlags = (old.mesonFlags or []) ++ [
            (pkgs.lib.mesonOption "default_library" "static")
          ];
        });

        commonEnv = {
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          # Use pre-built static libopus — bypass pkg-config to avoid dynamic linking
          LIBOPUS_STATIC = "1";
          LIBOPUS_NO_PKG = "1";
          LIBOPUS_LIB_DIR = "${libopusStatic}";
        };
      in {
        devShells.default = pkgs.mkShell (commonEnv // {
          buildInputs = [
            rustToolchain
            pkgs.cmake
            pkgs.pkg-config
            pkgs.libclang
            pkgs.llvmPackages.libclang
            pkgs.libopus
          ];

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
        });

        packages = {
          default = pkgs.rustPlatform.buildRustPackage (commonEnv // {
            inherit pname version;
            src = ./.;
            inherit cargoHash;

            nativeBuildInputs = with pkgs; [
              cmake
              pkg-config
              llvmPackages.libclang
            ];

            meta = with pkgs.lib; {
              description = "Text-to-speech MCP server powered by Fish Speech 1.5";
              license = licenses.mit;
            };
          });
        };
      }
    );
}
