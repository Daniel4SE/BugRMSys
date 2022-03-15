import logging
import pickle

from Constants import *

logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)


def load_df_compressed(file):
    return pd.read_csv(file, compression='gzip')


def save_df_compressed(file, df):
    return df.to_csv(file, compression='gzip', index=False)


def load_embeddings(file):
    try:
        with open(file, 'rb') as f:
            embeddings = pickle.load(f)
            logger.info(f'embeddings loaded from: {file}')
            return embeddings
    except FileNotFoundError:
        logger.info(f'embeddings file not found at: {file}')
        return {}

def save_embeddings(file, embeddings):
    with open(file, 'wb') as f:
        pickle.dump(embeddings, f, pickle.HIGHEST_PROTOCOL)
        logger.info(
            f'stored embeddings to {file}')


if __name__ == "__main__":
    print(True)
