# RamanGPT: corrected dataset and trained models

This folder holds the data and models behind the published results of

> F. M. Abel, J. Lee, C. R. Campbell & K. Choudhary, "RamanGPT maps bidirectionally between crystal structures and Raman spectra of inorganic materials", *Scientific Reports* (2026). https://doi.org/10.1038/s41598-026-79781-8

All numbers reported in the paper were produced from these files. The training code is in [atomgptlab/atomgpt](https://github.com/atomgptlab/atomgpt) (inverse model) and [atomgptlab/alignn](https://github.com/atomgptlab/alignn) (forward model), and the web application runs at https://atomgpt.org/raman.

## Older files in this repository use mispaired Raman tensors

The Computational Raman Database (CRD) stores the Raman tensors of 1,210 of these materials as a dictionary keyed by phonon-mode index. The compact export in `data/ramandb.json.zip` dropped those keys and lists the tensors in order. As a result, `02_preprocess_crd.py`, which pairs the i-th tensor with the i-th frequency, attaches each of those tensors to the wrong mode. The spectra in `alignn_dataset/`, the models in `runs/`, and the contents of `jae/checkpoint-888` and `jae/analyze-888` were built from that data. They are kept as a record of earlier work, but they do not reproduce the published results. Use the files in this folder instead.

## Data source and correction

Structures and Raman tensors come from CRD v1.0.1 (M. Bagheri & H.-P. Komsa, *Sci. Data* **10**, 80 (2023), https://github.com/mbagheri20/ramandb). The spectra here pair each tensor with its own mode and also include the corrected tensors that CRD v1.0.1 provides for 33 materials. Compared with the earlier data, 1,243 of the 5,092 forward targets changed (1,210 from the pairing fix and 33 from the v1.0.1 update). The inverse prompts were rebuilt from the same corrected modes.

Each spectrum is the orientation-averaged Raman activity, I = 45a² + 7γ², summed over all modes. Each mode is broadened with a Lorentzian of 3 cm⁻¹ full width at half maximum, and the result is sampled at 200 points from 50 to 1000 cm⁻¹ (`numpy.linspace(50, 1000, 200)`, 4.77 cm⁻¹ spacing) and normalized to a maximum of 1. These are activities, so they include no laser-wavelength or temperature factors.

## Contents

```
ramangpt_release/
├── forward/
│   ├── data/                    5,092 structures (mp-*.vasp, POSCAR format) and id_prop.csv
│   ├── ids_train_val_test.json  forward split: 4,073 train, 509 validation, 509 test
│   └── model/                   best_model.pt and config.json
├── inverse/
│   ├── alpaca_prop_train.json   4,574 training prompts
│   ├── alpaca_prop_test.json    508 test prompts
│   └── model/                   adapter config, tokenizer and training state (weights: see below)
└── SHA256SUMS
```

### Forward model (structure to spectrum)

- `forward/data/id_prop.csv` has one row per structure: the structure file name followed by the 200 spectrum values.
- `forward/ids_train_val_test.json` gives the split used in the paper. The structures were split in file order, and one structure (mp-540574) is in none of the three sets.
- `forward/model/` is the production model from the paper. It is an ALIGNN model with 6 ALIGNN layers and 6 edge-gated graph convolution layers, 256 hidden features, an 8.0 Å cutoff with 12 neighbors and a 200-value output, trained for 100 epochs with an L1 loss. It uses the pure-PyTorch ALIGNN (`alignn_atomwise_pure`) in atomgptlab/alignn and loads with `strict=True`.
- Test-set results (509 materials): the mean cosine similarity over spectrally active bins (target above 0.001) is 0.390, 53.8% of materials score 0.354 or higher, and the mean all-bin MAE is 0.017. If you regenerate the predictions from the structures, expect 53.4% instead of 53.8%. In 110 test structures the 12th and 13th neighbors are equally distant, so the neighbor list can differ from the one behind the paper's stored predictions. The mean stays at 0.390.

### Inverse model (spectrum and formula to structure)

- Each prompt record has the fields `instruction`, `input`, `id`, `peak_text` and `output`. The input gives the chemical formula and up to 20 of the most intense peaks with their relative intensities, for example:

  ```
  The chemical formula is: K3SbS4.
  The Raman spectrum shows main peaks at: 135.9cm-1(0.02), 193.2cm-1(0.22), 341.2cm-1(1.0), 350.8cm-1(0.2).
  Generate atomic structure description with lattice lengths, angles, coordinates and atom types.
  ```

  The output lists the lattice lengths (Å), the lattice angles (degrees), and then one line per atom with its element and fractional coordinates.
- The materials were split in the same file order into 4,574 training and 508 test materials, with no validation set. Every inverse test material is also in the forward test set.
- `inverse/model/` is the final checkpoint (step 1142) of a LoRA adapter with rank 16 and alpha 16 on the q, k, v, o, gate, up and down projections of `unsloth/mistral-7b-bnb-4bit`. Training started from the DiffractGPT adapter (`knc6/diffractgpt_mistral_chemical_formula` on Hugging Face) and ran for 2 epochs at a learning rate of 5e-5 with seed 3407. At inference the adapter is applied to the base model.
- The adapter weights, `adapter_model.safetensors` (168 MB), exceed GitHub's 100 MB file limit, so they are not stored in this repository. They are available from the corresponding author on request. Their SHA-256 checksum is `68ac443cfc30ed60f27c6c875893ef4f183d07a707702481ef7287b0658098f8`.
- Test-set results (508 materials): the reduced formula is preserved in 92.1% of cases, and the lattice-parameter MAEs are 1.13 Å (a), 1.15 Å (b) and 2.14 Å (c).

## Checksums

`SHA256SUMS` lists every data and model file in this folder. To verify them, run `shasum -a 256 -c SHA256SUMS` from inside `ramangpt_release/`.

The paper also names https://huggingface.co/datasets/knc6/ramangpt_dft as a release location. Check that any files you download there match these checksums.

## Citation

If you use these files, please cite the RamanGPT paper above and the Computational Raman Database:

> M. Bagheri & H.-P. Komsa, "High-throughput computation of Raman spectra from first principles", *Sci. Data* **10**, 80 (2023).
