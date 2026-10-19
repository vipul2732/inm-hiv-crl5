# Reference databases

`corum_humanComplexes.tsv` is the CORUM mammalian-complex table used for the manuscript benchmarks.

BioGRID, IntAct, STRING, and PDB coordinate files are not stored in this repository.

- BioGRID release used for the paper: **BIOGRID-ALL-5.0.253**. Direct physical interactions were the MI:0407 subset. Download the MITAB archive from the [BioGRID release archive](https://downloads.thebiogrid.org/BioGRID/Release-Archive/BIOGRID-5.0.253/) and follow the BioGRID license.
- CORUM: [https://mips.helmholtz-muenchen.de/corum/](https://mips.helmholtz-muenchen.de/corum/). The file in this folder is the copy used for the analyses. Please follow the CORUM license terms.
- PDB structures used for direct-contact validation are downloaded into `pdb/cache/` by `pdb/validate_direct_contact.py` when that script is re-run. The cache is not in git.

`script/validation/download_databases.sh` downloads STRING, IntAct, and BioGRID into this directory. Those full dumps are multi-gigabyte files and must stay on local disk.

The pair-level benchmark tables already deposited under `validation_pdb_and_databases/` record, for every modeled pair, whether it was supported by BioGRID, CORUM, or PDB. Those tables are sufficient to inspect the manuscript enrichment, recovery, and precision numbers without downloading the full databases.
