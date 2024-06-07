import tempfile
import zipfile

from pathlib import Path

import gdown

LINKS = {
    'preprocessed-croppad-1.5-128': 'https://drive.google.com/uc?id=1lBSA4lu1_aYWVzxZhCSiADyz6b7hNPxV',
    'preprocessed-croppad-1.5-128-distance': 'https://drive.google.com/uc?id=1GGBrps59CoBSVoynZPeNEUvwikQAfZAg',
}
DEFAULT_OUTPUT_PATH = Path('../data')


def download_data(
    output_dir: Path = DEFAULT_OUTPUT_PATH, dataset: str = 'preprocessed-croppad-1.5-128-distance'
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = str(Path(temp_dir) / 'temp_file.zip')
        gdown.download(LINKS[dataset], temp_file, quiet=False)

        with zipfile.ZipFile(temp_file) as zip_file:
            zip_file.extractall(output_dir)


if __name__ == '__main__':
    download_data()
