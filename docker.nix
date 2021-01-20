{ pkgs ? import <nixpkgs> {} }:

let
  app = (import ./poetry.nix { inherit pkgs; }).mkPoetryApplication {
    projectDir = ./.;
  };
in pkgs.dockerTools.streamLayeredImage {
  name = "wp5005";
  contents = [
    app.dependencyEnv
    # already in the closure and handy for `docker enter`:
    pkgs.bash
  ];
  config.Cmd = [ "/bin/sh" "-c" "/bin/uvicorn wp5005:app --port $PORT --host 0.0.0.0" ];
}
