{ pkgs }: {
  deps = [
    pkgs.python310Full
    pkgs.python310Packages.beautifulsoup4
    pkgs.python310Packages.lxml
  ];
}