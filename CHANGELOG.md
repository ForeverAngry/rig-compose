# Changelog

All notable changes to `rig-compose` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Versions are managed automatically by [release-plz](https://release-plz.dev/)
from [Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

## [0.1.0] - Unreleased

### Added

- Initial release of the composable agent kernel.
- Stateless `Skill` trait, transport-agnostic `Tool` trait, shared registries,
  generic agents, coordinator routing, workflows, instruction data, and
  in-process delegate support.
- Optional `manifest` feature for portable agent manifest loading.
