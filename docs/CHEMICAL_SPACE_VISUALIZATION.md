# Chemical-space visualization

Projection coordinates are exploratory views, not selection metrics.

* Property PCA shows standardized physicochemical descriptor differences.
* Structural UMAP shows local Morgan-fingerprint neighborhoods.
* Property UMAP shows nonlinear relationships among descriptors.
* TMAP is intended for large structural libraries and emphasizes neighborhood
  structure.

Every saved projection should retain its method, features, fingerprint settings,
parameters, seed, input hash and software versions. Changing a projection must
not change the molecular selection state.
