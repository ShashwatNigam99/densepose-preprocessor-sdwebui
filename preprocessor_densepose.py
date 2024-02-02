import torchvision # Fix issue Unknown builtin op: torchvision::nms
import cv2
import numpy as np
import torch
from einops import rearrange
from densepose import DensePoseMaskedColormapResultsVisualizer, _extract_i_from_iuvarr, densepose_chart_predictor_output_to_result_with_confidences
import os
import colorsys

N_PART_LABELS = 24
result_visualizer = DensePoseMaskedColormapResultsVisualizer(
    alpha=1,
    data_extractor=_extract_i_from_iuvarr,
    segm_extractor=_extract_i_from_iuvarr,
    val_scale = 255.0 / N_PART_LABELS
)
HSV_RANGE = (0.49,0.51)
HSV_VAL = 0.5
CCHANNELS = 3
COLREG = None
MAXCOLREG = 360 -1
# remote_torchscript_path = "https://huggingface.co/LayerNorm/DensePose-TorchScript-with-hint-image/resolve/main/densepose_r50_fpn_dl.torchscript"
torchscript_model = None
models_path = os.path.join(os.path.dirname(__file__), "models")
model_dir = os.path.join(models_path, "densepose")
print("Loading DensePose model from", model_dir)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CBLACK = 255
def make_white_masks(img):
    # img = cv2.imread(img_path)
    img[img>0] = 255
    return img

def deterministic_colours(n, lcol = None):
    """Generate n visually distinct & consistent colours as a list of RGB tuples.
    
    Uses the hue of hsv, with balanced saturation & value.
    Goes around the cyclical 0-256 and picks each /2 value for every round.
    Continuation rules: If pcyv != ccyv in next round, then we don't care.
    If pcyv == ccyv, we want to get the cval + delta of last elem.
    If lcol > n, will return it as is.
    """
    if n <= 0:
        return None
    pcyc = -1
    cval = 0
    if lcol is None:
        st = 0
    elif n <= len(lcol):
        # return lcol[:n] # Truncating the list is accurate, but pointless.
        return lcol
    else:
        st = len(lcol)
        if st > 0:
            pcyc = np.ceil(np.log2(st))
            # This is erroneous on st=2^n, but we don't care.
            dlt = 1 / (2 ** pcyc)
            cval = dlt + 2 * dlt * (st % (2 ** (pcyc - 1)) - 1)

    lhsv = []
    for i in range(st,n):
        ccyc = np.ceil(np.log2(i + 1))
        if ccyc == 0: # First col = 0.
            cval = 0
            pcyc = ccyc
        elif pcyc != ccyc: # New cycle, start from the half point between 0 and first point.
            dlt = 1 / (2 ** ccyc)
            cval = dlt
            pcyc = ccyc
        else:
            cval = cval + 2 * dlt # Jumps over existing vals.
        lhsv.append(cval)
    lhsv = [(v, 0.5, 0.5) for v in lhsv] # Hsv conversion only works 0:1.
    lrgb = [colorsys.hsv_to_rgb(*hsv) for hsv in lhsv]
    lrgb = (np.array(lrgb) * (CBLACK + 1)).astype(np.uint8) # Convert to colour uints.
    lrgb = lrgb.reshape(-1, CCHANNELS)
    if lcol is not None:
        lrgb = np.concatenate([lcol, lrgb])
    return lrgb

def apply_densepose(input_image, cmap=None, save_img=True, combined_mask=True  ):
    global torchscript_model
    if torchscript_model is None:
        model_path = os.path.join(model_dir, "densepose_r50_fpn_dl.torchscript")
        if not os.path.exists(model_path):
            print("Dense pose model doesnt exist, please check path ", model_path)
            exit()
        torchscript_model = torch.jit.load(model_path, map_location="cpu").to(DEVICE).eval()
    H, W  = input_image.shape[:2]

    hint_image_canvas = np.zeros([H, W], dtype=np.uint8)
    hint_image_canvas = np.tile(hint_image_canvas[:, :, np.newaxis], [1, 1, 3])
    input_image = rearrange(torch.from_numpy(input_image).to(DEVICE), 'h w c -> c h w')
    pred_boxes, corase_segm, fine_segm, u, v = torchscript_model(input_image)
    # breakpoint()
    extractor = densepose_chart_predictor_output_to_result_with_confidences
    densepose_results = [extractor(pred_boxes[i:i+1], corase_segm[i:i+1], fine_segm[i:i+1], u[i:i+1], v[i:i+1]) for i in range(len(pred_boxes))]

    if cmap=="viridis":
        result_visualizer.mask_visualizer.cmap = cv2.COLORMAP_VIRIDIS
        hint_image = result_visualizer.visualize(hint_image_canvas, densepose_results)
        hint_image = cv2.cvtColor(hint_image, cv2.COLOR_BGR2RGB)
        hint_image[:, :, 0][hint_image[:, :, 0] == 0] = 68
        hint_image[:, :, 1][hint_image[:, :, 1] == 0] = 1
        hint_image[:, :, 2][hint_image[:, :, 2] == 0] = 84
    else:
        result_visualizer.mask_visualizer.cmap = cv2.COLORMAP_PARULA
        # hint_image = result_visualizer.visualize(hint_image_canvas, densepose_results)
        # hint_image = cv2.cvtColor(hint_image, cv2.COLOR_BGR2RGB)
        
        # separate visualization for each part
        hint_images = result_visualizer.visualize_separate(hint_image_canvas, densepose_results)
        # hint_images = [cv2.cvtColor(hint_image, cv2.COLOR_BGR2RGB) for hint_image in hint_images]

    if save_img:
        for i, hint_image in enumerate(hint_images):
            cv2.imwrite("hint_image_{}.png".format(i), make_white_masks(hint_image))
    
    global COLREG
    if combined_mask:
        colors = deterministic_colours(2 * MAXCOLREG, COLREG)
        black = np.array([0, 0, 0])
        combined = np.zeros_like(hint_images[0])
        for i, hint_image in enumerate(hint_images):
            if i==4: #temporarily skipping the extra mask
                break
            print(colors[i])
            combined[np.where((hint_image!=[0,0,0]).all(axis=2))] =  [colors[i][2], colors[i][1], colors[i][0]] #  RGB --> RGB
        # combined[np.where((combined==[0,0,0]).all(axis=2))] =  [255,255,255]
        cv2.imwrite("combined_mask.png".format(i), combined)

    return hint_images


def unload_model():
    global torchscript_model
    if torchscript_model is not None:
        torchscript_model.cpu()
        
        
if __name__ == "__main__":
    img = cv2.imread("test.png")
    hint_image = apply_densepose(img)
    unload_model()