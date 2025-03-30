This project focuses on developing a deep learning solution to harmonize multi-resolution satellite imagery (Landsat 8 and Sentinel-2) for geological and environmental monitoring. The implemented CNN architecture bridges resolution gaps between datasets, enabling enhanced analysis of mining areas near Khabarovsk, Russia. Developed in collaboration with the Institute of Mining, Far Eastern Branch of the Russian Academy of Sciences.
Key Features

    Cross-Sensor Compatibility: Processes 400+ multi-spectral images from Landsat 8 (30m resolution) and Sentinel-2 (10m/20m).

    Resolution Harmonization: CNN-based model to approximate high-resolution features from lower-resolution inputs.

    Geolocalized Focus: Targets quarries in the Khabarovsk region for precision in industrial/environmental impact studies.

Methodology

    Data Pipeline:

        Preprocessed TIFF/GeoTIFF images (atmospheric correction, band alignment, normalization).

        Pixel-wise registration to align Sentinel-2 and Landsat 8 datasets.

    CNN Architecture:

        Custom U-Net-inspired model with residual blocks for feature preservation.

        Loss function combining MSE (structural fidelity) and SSIM (textural similarity).

    Validation:

        Quantitative metrics: PSNR (>32 dB), SSIM (>0.89) on test sets.

        Qualitative assessment by geologists at the Institute of Mining.

Dataset

    Sources: ESA Copernicus (Sentinel-2), USGS EarthExplorer (Landsat 8).

    Scope: 2020-2023 imagery covering 12 mining sites near Khabarovsk.

    Preprocessing: GDAL for georeferencing, OpenCV for histogram matching.

Results

    Achieved 93% spatial consistency in approximating Sentinel-2-like resolution from Landsat 8 inputs.

    Reduced data preprocessing time for the Institute’s analysts by 40% compared to manual interpolation methods.
