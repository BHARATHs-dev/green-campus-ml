import numpy as np
from typing import Dict, Optional


def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    ndvi = (nir - red) / (nir + red + 1e-10)
    return np.clip(ndvi, -1.0, 1.0)


def calculate_evi(red: np.ndarray, nir: np.ndarray, blue: np.ndarray) -> np.ndarray:
    ndvi = calculate_ndvi(red, nir)
    evi = 2.5 * ndvi / (nir + 6 * red - 7.5 * blue + 1.0)
    return np.clip(evi, -1.0, 1.0)


def calculate_savi(nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> np.ndarray:
    savi = ((nir - red) / (nir + red + L)) * (1 + L)
    return np.clip(savi, -1.0, 1.0)


def calculate_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    ndwi = (green - nir) / (green + nir + 1e-10)
    return np.clip(ndwi, -1.0, 1.0)


def calculate_msi(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    msi = swir / (nir + 1e-10)
    return np.clip(msi, 0.0, 10.0)


def apply_spectral_indices(bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    indices = {}

    if 'B4' in bands and 'B8' in bands:
        indices['ndvi'] = calculate_ndvi(bands['B4'], bands['B8'])

    if 'B4' in bands and 'B8' in bands and 'B2' in bands:
        indices['evi'] = calculate_evi(bands['B4'], bands['B8'], bands['B2'])

    if 'B4' in bands and 'B8' in bands:
        indices['savi'] = calculate_savi(bands['B8'], bands['B4'])

    if 'B3' in bands and 'B8' in bands:
        indices['ndwi'] = calculate_ndwi(bands['B3'], bands['B8'])

    if 'B8' in bands and 'B11' in bands:
        indices['msi'] = calculate_msi(bands['B8'], bands['B11'])
    elif 'B8' in bands and 'B12' in bands:
        indices['msi'] = calculate_msi(bands['B8'], bands['B12'])

    return indices


def normalize_band(band: np.ndarray, method: str = 'minmax') -> np.ndarray:
    if method == 'minmax':
        band_min = band.min()
        band_max = band.max()
        if band_max - band_min > 1e-10:
            return (band - band_min) / (band_max - band_min)
        return band - band_min
    elif method == 'zscore':
        mean = band.mean()
        std = band.std()
        if std > 1e-10:
            return (band - mean) / std
        return band - mean
    else:
        raise ValueError(f"Unknown normalization method: {method}")
