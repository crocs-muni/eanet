# eanet
ANN version of EACirc

## Running notes

### Network

The main scripts are `eanet_ann.py` for basic aritfitial neural network and `eanet_cnn.py` for convoluted version, where the first layer groups input into bytes.

Both scripts has same interface (and almost same code (it could be nice to merge them together with one more argument).

```
> python eanet_ann.py --in_a REFERENCE.bin --in_b TESTED.bin --tv_size BYTE_SIZE_OF_TEST_VECTOR
```

Where `TESTED.bin` is binary file for test, `REFERENCE.bin` is binary file with truly random data (or strong PRNG). The `BYTE_SIZE_OF_TEST_VECTOR` sets the width of first layer and it should correspond to the source data (AES has 16 B block, so set at least 16 B).

The output is:

```
> Using TensorFlow backend.
> Preparing test vectors
> Training started
> Epoch 1/100
> 2500/2500 [==============================] - 0s - loss: 0.6973 - acc: 0.5084
...
> Epoch 100/100
> 2500/2500 [==============================] - 0s - loss: 0.0372 - acc: 1.0000     
> Trained
> Preparing final test vectors
> 12500/12500 [==============================] - 0s
> Score = [0.042080752551555634, 0.99936002492904663]
```

Simplified, the `acc` means what ratio of correct guesses hat the ANN on the data during given batch. The `loss` says, how fast the ANN learns - lower numbers imply smaller changes - fine grained fine-tuning of the values.

The final evaluation is done on fresh data (and much bigger batch). The score is tuple of `loss` and `acc`. The second value for random data should act accordingly to binomial distribution with `n=12500` and `p=0.5`. The `p-value` of this computation would be 1. Even `acc=0.55` is significant for that big batch.

### Automation

The script `run_ann.py` enables running scripts `eanet_ann.py` and `eanet_cnn.py` on multiple binary files in sequence. It expects you have reference file named `in_b.bin` for running ANN scripts. Then it runs given script over all binary files in current directory. If the binary files have naming convention `NAME_bSIZE.bin` for example `AES_r3_b16.bin`, it will pass the corresponding block size to the ANN script (in the example, it will set `--tv_size 16`). If the size is not set, the default value is 16 B.

```
> python3 run_ann.py --script SCRIPT_NAME
```

The output is just persistent keras info, that we are using TensorFlow backend. But the script creates `SCRIPT_NAME_result.json` result file and also loggs the computation to `ann_runner.log`.

### Processing the results

The `eanet_ann.py_result.json` can be processed by `process.py` script, that creates figures of the learning process of the ANN computations and highlights the final values. It has no arguments, but the processed files have to be named `result_ann.json` and `result_cnn.json`. You also need to have directories: `ann` and `cnn`, where the figures are stored.

