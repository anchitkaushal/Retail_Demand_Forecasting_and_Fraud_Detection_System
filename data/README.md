Data folder layout
- raw/: original immutable datasets (source of truth)
- interim/: intermediate extracts and cleaned partial files
- processed/: final feature tables ready for modeling
- external/: external datasets such as holidays, weather, promotions

Do NOT commit large raw data files into the repository. Keep data pointers or small samples instead.
