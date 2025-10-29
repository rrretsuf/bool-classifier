from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
from typing import List, Optional

class Embedder:
    # model_name: str = "BAAI/bge-m3" je tukaj default za vsak slučaj (ai gen code sori)
    def __init__(self, model_name: str = "BAAI/bge-m3", device: Optional[str] = None):
        """
        init se embedder. če obstaja lokalna kopija modela, jo naloži, sicer potegne iz interneta.
        """
        self.model_name = model_name
        self.device = device
        self.model: Optional[SentenceTransformer] = None
    
    def _lazy_init(self):
        """
        zamujeno nalaganje modela, samo ko ga res rabiš.
        """
        if self.model is None: 
            print(f"[embeddings] Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        pretvori seznam besedil v numpy matriko embeddings.
        """
        self._lazy_init()
        # sentance-transformers že zna batchirati, ampak mi ga ovijemo za varnost
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=64,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return np.asarray(embeddings)

    def save(self, path: str | Path): 
        """
        shrani model
        """
        self._lazy_init()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        print(f"[embeddings] Saving model to {path}")
        self.model.save(str(path))

    def load(self, path: str | Path):
        """
        naloži že shranjen lokalni model iz diska.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Embedder path does not exist: {path}")
        print(f"[embeddings] Loading local model from {path}")
        self.model = SentenceTransformer(str(path), device=self.device)