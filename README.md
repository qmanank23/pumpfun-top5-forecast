# pumpfun-top5-forecast

Forecasts the 5 largest new pump.fun coins by market cap at each of seven
answer reads, with honest probabilities and size bins, for make.fun's
"Predict the top 5 new Solana coins by market cap, 24 hours out" bounty.

## How it works

`main.ts` (Node 22, standard library only, no network) reads the house's
10-minute market-cap snapshots from `/input`, then runs a horizon-averaged
Monte Carlo over the candidate chart: each coin's future log10 market cap is
a shrunk-drift normal mixed with a "death" component for young bonding-curve
coins, sampled at every answer-read horizon from `task.json` with a fixed
seed. Place probabilities and size-bin probabilities come from the same
draws, with a large unnamed reserve per place for coins born after the cut,
which no history can name.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

The forecast is written to `out/forecast.json`. Two runs on the same input
produce byte-identical output.
