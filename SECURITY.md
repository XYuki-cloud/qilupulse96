# Security policy

## Reporting

Do not open a public issue for secrets, credential exposure, private data,
path leaks that reveal a real machine, or a vulnerability that could affect a
deployed data pipeline. Contact the repository maintainer privately through
the GitHub security reporting mechanism or the maintainer's verified contact
channel.

Include the affected commit or file, a minimal reproduction, impact, and any
safe mitigation. Do not include API keys, cookies, raw market files, or model
weights in the report.

## Release checks

Every public release should scan the working tree and Git history for secrets,
large data files, absolute paths, and ignored runtime artifacts. A successful
test run alone is not a security review.
