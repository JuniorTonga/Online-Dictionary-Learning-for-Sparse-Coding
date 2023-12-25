import numpy as np
from .utils import show_imgs,l2_error, insert_text_on_image


def infer_img(img, dict_learner):
    h, w = img.shape[:2]
    if len(img.shape) == 3:
        img = img.flatten()[np.newaxis, :]
    img_transformed = dict_learner.transform(img)
    img_hat = img_transformed @ dict_learner.components_
    img_hat = np.reshape(img_hat, (1,h, w,3))[0]
    return img_hat


def forge_and_reconstruct(img, dict_learner, text, bottom_left=(10, 30), color=(1, 0, 0), display=True):
    forged_img = insert_text_on_image(
        img, text, scale=0.8,bottom_left=bottom_left, color=color,line_type=2)

    img_hat = infer_img(img, dict_learner)
    forged_img_hat = infer_img(forged_img, dict_learner)

    if display:
        l2_original_forge = l2_error(img, img_hat)
        title_originals = f"L2 error between original and its reconstruction: {l2_original_forge:.3f}\n"

        l2_reconstruction_bt_img = l2_error(img , forged_img_hat)
        title_reconstructs = f"L2 error between the original image and its reconstruction when altered with text : {l2_reconstruction_bt_img:.3f}"

        suptitle = title_originals + title_reconstructs
        show_imgs([img, forged_img, img_hat, forged_img_hat], ["Original", "original_with_text",
                  "Original reconstructed", "original_with_text_reconstructed"],
                  suptitle=suptitle, figsize=(15, 4))

    return forged_img, img_hat, forged_img_hat

