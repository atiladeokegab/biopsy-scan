![Can your study find it?](docs/assets/banner.png)

# Can your study find it?

A GPU engine that re-images thousands of simulated spatial-biomarker studies cell by cell from real head and neck cancer tissue (UPMC CODEX, Charville), so a lab can see, before it images a single slide, which study designs (patients, regions per patient, panel, imaging error) can actually find a survival-linked spatial pattern like Helsinki's stroma finding. The reshaping (nesting, region pools, one shared null threshold per design point) and a speedup ladder that separates algorithm from hardware against the best adequate CPU method.

## What it does

A GPU engine that re-images thousands of simulated spatial-biomarker studies cell by cell from real head and neck cancer tissue (UPMC CODEX, Charville), so a lab can see, before it images a single slide, which study designs (patients, regions per patient, panel, imaging error) can actually find a survival-linked spatial pattern like Helsinki's stroma finding. The reshaping (nesting, region pools, one shared null threshold per design point) and a speedup ladder that separates algorithm from hardware against the best adequate CPU method.

## Demo

The working demo will be added during the build.

## Screenshots

Screenshots will be added after the first working demo.

## Architecture

See [the product plan](IDEA.md) for the architecture.

## Run locally

```sh
gh repo clone atiladeokegab/sota-parallelism-nov21
cd sota-parallelism-nov21
make dev
```

## Team

- [Atilade](https://github.com/atiladeokegab)
- [Faisal](https://github.com/FHL-08)
