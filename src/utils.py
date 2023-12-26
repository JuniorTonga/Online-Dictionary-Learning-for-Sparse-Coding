import cv2
import numpy as np
import matplotlib.pyplot as plt

def insert_text_on_image(img, texts=[], scale=0.7, bottom_left=(10, 100), color=(255, 0, 0), line_type=2):
    
    # Récupérer les dimensions de l'image
    h, w = img.shape[:2]

    # Calculer le ratio de redimensionnement en fonction de la plus grande dimension
    ratio = float(max(200 / h, 200 / w))

    # Redimensionner l'image si nécessaire
    if ratio > 1.0:
        img = cv2.resize(img, (int(ratio * w), int(ratio * h)))

    # Ajouter chaque ligne de texte à l'image
    for idx, text in enumerate(texts):
        # Calculer la position verticale pour chaque ligne de texte
        text_position = (bottom_left[0], bottom_left[1] + 25 * idx)

        # Ajouter le texte à l'image
        cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, line_type)

    # Revenir à la taille d'origine si l'image a été redimensionnée
    if ratio > 1.0:
        img = cv2.resize(img, (w, h))

    return img

def show_imgs(imgs, titles, figsize=(30, 30), suptitle=None):
    fig, axes = plt.subplots(ncols=len(imgs), figsize=figsize)
    if len(imgs) == 1:
        img=np.clip(imgs[0], 0, 1)
        axes.imshow(img)
        axes.set_title(titles[0])
    else:
        for idx, img in enumerate(imgs):
            img=np.clip(img, 0, 1)
            axes[idx].imshow(img)
            axes[idx].set_title(titles[idx])
    if suptitle is not None:
        fig.suptitle(suptitle)
    plt.show()


def l2_error(img, img_hat):
    return np.linalg.norm(img - img_hat)



def show_atoms_img(atoms: np.ndarray, figsize=(15, 8)):
    n_atoms = atoms.shape[0]
    n_imgs_per_row = 4
    nrows, ncols = int(np.ceil(n_atoms / n_imgs_per_row)
                       ), min(n_imgs_per_row, n_atoms)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx < n_atoms:
            ax.imshow(atoms[idx])
        else:
            ax.axis('off')
    fig.suptitle("Atoms")
    plt.show()


def show_dictionary_atoms_img(dict_learner, color=False, atom_h=150, atom_w=150, figsize=(15, 8)):
    if color:
        new_shape = (dict_learner.n_components, atom_h, atom_w, 3)
    else:
        new_shape = (dict_learner.n_components, atom_h, atom_w)
    atoms = np.reshape(dict_learner.components_, new_shape)

    atoms = (atoms - np.min(atoms)) / (np.max(atoms) - np.min(atoms))
    show_atoms_img(atoms, figsize=figsize)




def plot_reconstruction_error(
        ax, times, reconstruction_errors, label_name="Test image", label_values=[]):

    # Plot the reconstruction error
    for idx, single_error_curve in enumerate(reconstruction_errors):
        label = f"{label_name} {label_values[idx]}" if len(
            label_values) > 0 else f"{label_name} {idx}"
        x_axis = times if type(times) == list else times[idx]
        ax.loglog(x_axis, single_error_curve, label=label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reconstruction error")
    ax.set_title("Reconstruction error through time")
    ax.legend()

def plot_reconstruction_error_(
        times, reconstruction_errors, label_name="Test image", label_values=[]):
    _, ax0 = plt.subplots(ncols=1, figsize=(15, 5))

    plot_reconstruction_error(
        ax0, times, reconstruction_errors, label_name, label_values)

    plt.show()
