import torch
import torch.nn.functional as F
from src.regression import TwoWordTensorRegression, OneWordTensorRegression, CPTensorRegression
import spacy
from sentence_transformers import SentenceTransformer
import os

class ModelBank(object):
    def __init__(self, model_locations: str):
        self.reference_caches: dict[str, list] = dict()
        self.model_caches: dict[tuple[str, str], torch.nn.Module] = dict()
        self.BERT_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        self.nlp: spacy.load = None

        self.model_locations = model_locations

    def load_reference(self, lang_type: str, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Reference file for {lang_type} not found at {file_path}")
        else:
            print(f"Loading reference list for {lang_type} from {file_path}")
        
        with open(file_path, "r") as f:
            self.reference_caches[lang_type] = f.read().splitlines()
        return self.reference_caches[lang_type]

    def retrieve_BERT(self, text: str) -> torch.Tensor:
        """
        Retrieves the BERT embedding for the given text.
        """
        word_embedding = self.BERT_model.encode(text, convert_to_tensor=True)
        word_embedding = word_embedding.cpu().numpy().reshape(1, -1)

        return torch.from_numpy(word_embedding)

    def ann(self, target: str, lang_type: str):
        
        candidates: list[str] = []
        if lang_type in self.reference_caches:
            candidates = self.reference_caches[lang_type]
        else:
            file_directory = f"{self.model_locations}/{lang_type}"
            if not os.path.exists(file_directory):
                raise FileNotFoundError(f"Directory for {lang_type} not found at {file_directory}")
            else:
                candidates = os.listdir(file_directory)
        
        #evaluation:
        max_score = float('-inf')
        best_candidate = None
        target_embedding = self.retrieve_BERT(target)

        for candidate in candidates:
            candidate_embedding = self.retrieve_BERT(candidate)

            score = F.cosine_similarity(target_embedding, candidate_embedding, dim=1).item()
            
            print(f"Comparing '{target}' with '{candidate}': score = {score}")
            
            if score > max_score:
                max_score = score
                best_candidate = candidate
                if score >= 0.99:
                    break
        
        print(f"Best candidate: '{best_candidate}' with score {max_score}")

        return best_candidate, max_score if best_candidate else "No suitable candidate found"

    def load_model(self, directory: str, model_name: str, n: int) -> torch.nn.Module:
        """
        load regression model from the given path.
        """
        model_path = f"{self.model_locations}/{directory}/{model_name}"
        
        model = CPTensorRegression([384 for _ in range(n)], 384, 100)
        #model = OneWordTensorRegression(384, 384)

        print("MODEL TYPE ", type(torch.load(model_path, weights_only=True)))
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        return model

    def load_ann(self, ID: tuple[str, str], n: int) -> torch.nn.Module:
        """
        load ANN model from the given path.
        """
        if ID in self.model_caches:
            model = self.model_caches[ID]
        else:
            try:
                model = self.load_model(ID[1], ID[0], n=n)
            except:
                try:
                    print(f"File {self.model_locations}/{ID[1]}/{ID[0]} not found, checking lemma...")
                    word = ID[0]
                    if self.nlp is None:
                        print("spaCy model uninitialized.")
                        raise ValueError("spaCy model uninitialized.")
                    else:
                        doc = self.nlp(word)
                        
                        for token in doc:
                            word = token.lemma_
                            print(f"lemma: {word}")
                    
                        model = self.load_model(ID[1], word, n=n)
                except:
                    print(f"Model for lemmatized form of {ID[0]} not found, finding nearest neightbor...")
            
                    word = ID[0]
                    nearest_name, _ = self.ann(ID[0], ID[1])
                    model = self.load_model(ID[1], nearest_name, n = n)
            
            self.model_caches[ID] = model

        return model
    
    def set_nlp(self, nlp: spacy.load):
        self.nlp = nlp
    


if __name__ == "__main__":
    """
    EXAMPLE USAGE:
    """
    cache = ModelBank("/mnt/ssd/user-workspaces/aidan-svc/CCD_tensor_training")

    nlp = spacy.load("en_core_web_trf")
    cache.set_nlp(nlp)
    model = cache.load_ann(("among", "prep_model"), n=1)

    print(type(model))