Run `python preprocessor_densepose.py` for generating mask.

Input: `test.png`

Output mask: `combined_mask.png`

Mask can be uploaded [regional prompter extension on SD-webui](https://github.com/hako-mikan/sd-webui-regional-prompter?tab=readme-ov-file#mask-regions-aka-inpaint-experimental-function).

Place model inside `./models/densepose/`.

Model can be found [here](https://huggingface.co/LayerNorm/DensePose-TorchScript-with-hint-image/blob/main/densepose_r50_fpn_dl.torchscript)
