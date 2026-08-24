# Model release policy

The repository publishes the QiluPulse-96 model interface and bundle format,
not a production checkpoint by default.

A separate model release is allowed only when all checks below have evidence:

- the publisher owns or is authorized to redistribute the weights;
- every training dataset and derived feature is permitted for redistribution;
- the weight file contains no private or user-identifying data;
- the model manifest names the architecture, feature schema, training cutoff,
  weather kind, and parameter checksum;
- the distribution channel and file size are appropriate;
- the README states that the model is not a guarantee of market performance;
- a clean CPU load test verifies the checksum and output schema.

Until then, use the synthetic and test fixtures only. Do not place `.pt`,
`.pth`, `.ckpt`, `.onnx`, `.npz`, or `.safetensors` files in Git.
