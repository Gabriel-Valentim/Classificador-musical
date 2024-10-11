import skimage
import numpy as np
import math


def rol(n, rotations, width=8):
    return ((n << rotations) & ((1 << (width - 1)) - 1)) | (n >> (width - rotations))


def get_lbp_histogram(im, p, r):
    # calculate uniform lbp for image
    lbp = skimage.feature.local_binary_pattern(im, p, r, method="nri_uniform")

    # store number of occurences in histogram
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))

    return hist


def get_rlbp_histogram(im, p, r):
    # calculate regular lbp for image
    lbp = skimage.feature.local_binary_pattern(im, p, r, method="default")

    # store number of occurences in histogram
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp, density=False, bins=n_bins, range=(0, n_bins))

    rlbp_hist = np.zeros((n_bins,), lbp_hist.dtype)

    # 11010000

    # for all bit patterns k of length p:
    # find all occurences of bit patterns 101 and 010 in k
    # print all occurences
    for k in range(n_bins):
        # check if k is non-uniform
        # k is non-uniform if it has more than 2 bit flips

        flips = k ^ rol(k, p - 1, p)
        cnt = flips.bit_count()

        if cnt > 2:
            different = False
            for rot in range(p):
                krot = rol(k, rot, p)
                if (krot & 0b111) == 0b101:
                    rrot = krot | 0b111
                    tgtk = rol(rrot, p - rot, p)
                    rlbp_hist[tgtk] += lbp_hist[k]
                    different = True
                elif (krot & 0b111) == 0b010:
                    rrot = krot ^ (krot & 0b111)
                    tgtk = rol(rrot, p - rot, p)
                    rlbp_hist[tgtk] += lbp_hist[k]
                    different = True
            if not different:
                rlbp_hist[k] += lbp_hist[k]
        else:
            rlbp_hist[k] += lbp_hist[k]

    # finally, restrict histogram to only uniform patterns
    # group non-uniform patterns in extra bucket
    # there are C(p, 2) * 2 + 2 uniform patterns + 1 non-uniform bucket
    final_hist = np.zeros((math.comb(p, 2) * 2 + 2 + 1,), lbp_hist.dtype)

    occurences = sum(rlbp_hist)
    final_hist[0] = rlbp_hist[0]
    final_hist[1] = rlbp_hist[2**p - 1]
    i = 2

    ALL_ONES = 2**p - 1
    for l in range(1, p):
        num = (ALL_ONES ^ (ALL_ONES << l)) & ALL_ONES
        for k in range(p):
            final_hist[i] = rlbp_hist[num]
            i += 1
            num = rol(num, 1, p)

    final_hist[i] = sum(final_hist[0:i]) - occurences
    final_hist = final_hist.astype(np.float32) / sum(final_hist)

    return final_hist




-------------------------------------------


import math
from pathlib import Path
import librosa
import librosa.feature
import librosa.feature.rhythm
import numpy as np

import skimage

from . import lbp, lpq
from . import lbp, lpq

# rhythm patterns
# add cwd to module path
import os
import sys

sys.path.append(os.getcwd())
from vendor.rp_extract.rp_extract import rp_extract


def fv_lbp_rp_ex(audio_path: Path, spectrogram_path: Path):
    img = skimage.io.imread(spectrogram_path, as_gray=True)

    lbp_hist = lbp.get_lbp_histogram(img, p=8, r=2)

    # use rhythm pattern extraction from vendor/rp_extract
    x, fs = librosa.load(audio_path, sr=44100, mono=True)
    rp = rp_extract(x, fs, extract_rp=True)["rp"]

    return np.hstack((lbp_hist, rp))


def fv_lbp_ex(audio_path: Path, spectrogram_path: Path):
    img = skimage.io.imread(spectrogram_path, as_gray=True)

    lbp_hist = lbp.get_lbp_histogram(img, p=8, r=2)
    return np.hstack((lbp_hist,))


def fv_rp_ex(audio_path: Path, spectrogram_path: Path):
    # use rhythm pattern extraction from vendor/rp_extract
    x, fs = librosa.load(audio_path, sr=44100, mono=True)
    rp = rp_extract(x, fs, extract_rp=True)["rp"]
    return np.hstack((rp,))


def fv_lpq_ex(audio_path: Path, spectrogram_path: Path):
    img = skimage.io.imread(spectrogram_path, as_gray=True)

    lpq_hist = lpq.get_lpq_histogram(img)
    return np.hstack((lpq_hist,))


def fv_glcm_ex(audio_path: Path, spectrogram_path: Path):
    img = skimage.io.imread(spectrogram_path, as_gray=True)
    mats = skimage.feature.graycomatrix(
        img, [1, 2], [0, math.pi / 4, math.pi / 2, 3 * math.pi / 4], levels=256
    )
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    vals = sum(
        (skimage.feature.graycoprops(mats, prop).flatten().tolist() for prop in props),
        [],
    )
    return np.hstack((vals,))


def fv_mfcc_ex(audio_path: Path, spectrogram_path: Path):
    y, fs = librosa.load(audio_path, sr=44100, mono=True)
    mfccs = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=13)

    mfccs = np.vstack(
        (
            np.mean(mfccs, axis=1),
            np.median(mfccs, axis=1),
            np.max(mfccs, axis=1),
            np.min(mfccs, axis=1),
            np.std(mfccs, axis=1),
        )
    )

    return np.hstack((mfccs.flatten(),))


def fv_glcm_mfcc_ex(audio_path: Path, spectrogram_path: Path):
    glcm = fv_glcm_ex(audio_path, spectrogram_path)
    mfccs = fv_mfcc_ex(audio_path, spectrogram_path)

    return np.hstack(
        (
            glcm,
            mfccs,
        )
    )


def fv_lbp_mfcc_ex(audio_path: Path, spectrogram_path: Path):
    lbp = fv_lbp_ex(audio_path, spectrogram_path)
    mfccs = fv_mfcc_ex(audio_path, spectrogram_path)

    return np.hstack(
        (
            lbp,
            mfccs,
        )
    )


def fv_lpq_mfcc_ex(audio_path: Path, spectrogram_path: Path):
    lpq = fv_lpq_ex(audio_path, spectrogram_path)
    mfccs = fv_mfcc_ex(audio_path, spectrogram_path)

    return np.hstack(
        (
            lpq,
            mfccs,
        )
    )


def fv_lbp_mfcc_glcm_ex(audio_path: Path, spectrogram_path: Path):
    lbp = fv_lbp_ex(audio_path, spectrogram_path)
    mfccs = fv_mfcc_ex(audio_path, spectrogram_path)
    glcm = fv_glcm_ex(audio_path, spectrogram_path)

    return np.hstack((lbp, mfccs, glcm))


fv_lbp_rp = ("lbp-rp", fv_lbp_rp_ex)
fv_lbp = ("lbp", fv_lbp_ex)
fv_rp = ("rp", fv_rp_ex)
fv_lpq = ("lpq", fv_lpq_ex)
fv_glcm = ("glcm", fv_glcm_ex)
fv_mfcc = ("mfcc", fv_mfcc_ex)
fv_glcm_mfcc = ("glcm-mfcc", fv_glcm_mfcc_ex)
fv_lbp_mfcc = ("lbp-mfcc", fv_lbp_mfcc_ex)
fv_lpq_mfcc = ("lpq-mfcc", fv_lpq_mfcc_ex)
fv_lbp_mfcc_glcm = ("lbp-mfcc-glcm", fv_lbp_mfcc_glcm_ex)