# Third-party and inherited material

The public source tree does not vendor the runtime dependencies below. They
are installed under their own licenses, and those licenses are not replaced by
the Apache-2.0 license for QiluPulse-96. A wheel or source distribution must
continue to expose the package metadata and notices supplied by its
dependencies.

## Direct dependencies

| Package | Use in this project | Upstream license / notice |
| --- | --- | --- |
| `holidays` | China calendar features | MIT |
| `matplotlib` | diagnostics and report plots | Matplotlib License; bundled notices are part of the package |
| `numpy` | numerical arrays | BSD-3-Clause |
| `openpyxl` | operator Excel import | MIT |
| `pandas` | tabular time-series processing | BSD-3-Clause |
| `pyarrow` | Parquet input/output | Apache-2.0 |
| `pvlib` | solar-position and clear-sky features | MIT |
| `requests` | Open-Meteo HTTP adapter | Apache-2.0 |
| `scikit-learn` | scaling, PCA, and validation helpers | BSD-3-Clause |
| `torch` | QiluPulse-96 model runtime | BSD-3-Clause; see the installed distribution notices |

The build backend (`hatchling`) and development test runner (`pytest`) are
also third-party packages and retain their upstream licenses. Transitive
packages are not copied into this repository; inspect the lock file and the
installed distribution metadata when preparing a redistributable binary.

## External data and service boundary

Open-Meteo is an external service. The source code records the request and
forecast-response evidence it receives, but it does not grant a license to
redistribute service responses. Market data, weather snapshots, calibration
ledgers, and operator-provided workbooks are deliberately excluded. No data
license is granted by this repository.

## Inherited-code gate

The files listed as `OWNER_CONFIRMATION_REQUIRED` in
`docs/PROVENANCE.md` must not be published under Apache-2.0 until their author,
source, license, and retained notices are confirmed. If a module was copied or
adapted from another project, add its source URL/revision, copyright notice,
license, and the reason it is compatible with this repository here before
publication.
