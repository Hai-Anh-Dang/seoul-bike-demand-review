# Raw data provenance

Dataset: **Seoul Bike Sharing Demand**. UCI Machine Learning Repository (2020). DOI: [10.24432/C5F62R](https://doi.org/10.24432/C5F62R).

Source metadata: [UCI dataset page](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand). License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Source metadata and licence checked on 7 October 2026.

`SeoulBikeData.csv` is a byte-for-byte copy of the dataset supplied with the original university analysis. Its 8,760 records cover one calendar year of Seoul system hourly rentals and weather. The SHA-256 is recorded in `provenance.json` and checked by the notebook. This hash establishes the bundled file identity; it is not claimed to be independently compared with a fresh UCI download.

The raw file is not edited. The notebook reads the original encoding (`cp1252`), preserves the raw table, creates readable analytical names in a derived table, and parses the day/month/year date format explicitly. The visibility field is in 10-metre units. Operating status and all zero counts are preserved. No outlier deletion or missing-value imputation is applied.

The project is self-contained: the notebook reads this file from `data/`. Dataset terms are also documented in [dataset licence and attribution](#dataset-licence-and-attribution).

## Dataset licence and attribution

**Seoul Bike Sharing Demand** [Dataset]. (2020). UCI Machine Learning Repository. DOI: [10.24432/C5F62R](https://doi.org/10.24432/C5F62R).

Source: [UCI dataset page](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand). UCI identifies the dataset licence as [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). That licence applies to the dataset, with attribution required under its terms.

`SeoulBikeData.csv` retains its supplied bytes. The notebook creates derived analytical columns and outputs without editing that raw file. `provenance.json` records the source and SHA-256. Dataset licensing does not grant a blanket software licence for this project; see [attribution and use of AI](#attribution-and-use-of-ai).

## Attribution

The analytical starting point was a group assignment by Thanh Nguyen, Anh Dang, Nhat Nam Phan and Khoa Nguyen. This Python adaptation retains the model families and dataset while correcting evaluation issues and adding baselines, diagnostics and clearer limits.
