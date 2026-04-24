# Weather Files

This directory must contain one weather file per simulation location in `.mos` format, as required by TEASER for OpenModelica simulations. Weather files are not included in this repository due to licensing restrictions. You are free to use weather data from any source; [AixWeather](https://github.com/RWTH-EBC/AixWeather) — available as a Python library and as a [web application](https://aixweather.eonerc.rwth-aachen.de/) — can convert a range of common weather data formats to `.mos`.

## Recommended dataset: DWD Test Reference Years (TRY 2017)

We recommend using the **[DWD Test Reference Year (TRY) 2017](https://www.dwd.de/DE/leistungen/testreferenzjahre/testreferenzjahre.html)** dataset, published by the German Weather Service (Deutscher Wetterdienst, DWD).

A Test Reference Year (TRY) is a synthetic annual weather dataset constructed to be statistically representative of the long-term climate at a given location. The DWD TRY 2017 dataset covers Germany at a spatial resolution of 1 km × 1 km. It is provided in three temporal variants — **annual**, **summer**, and **winter** — and two climate scenarios: **present** (based on the reference period 1995–2012) and **future** (projected climate conditions around 2045). This yields six dataset variants in total per grid cell.

> **Licensing note:** The TRY 2017 dataset is subject to the DWD terms of use and cannot be redistributed. It is freely available for download from the DWD [Climate Consulting Module (Klimaberatungsmodul)](https://kunden.dwd.de/obt/).

## Weather Data Used in This Study

For the study presented in this repository, the **annual, present-climate** TRY 2017 variant was selected for six German cities (Berlin, Hamburg, Düsseldorf, Munich, Frankfurt, and Leipzig), consistent with the full-year simulation period of 365 days. To reproduce this setup, download the TRY 2017 grid cell corresponding to each city from the Climate Consulting Module, convert them to `.mos` format using AixWeather, and place the files in this directory with the following filenames (matching the default `config.yaml`):

```
berlin.mos
hamburg.mos
dusseldorf.mos
munich.mos
frankfurt.mos
leipzig.mos
```

Once all six files are in place, the pipeline can be executed from the project root.
