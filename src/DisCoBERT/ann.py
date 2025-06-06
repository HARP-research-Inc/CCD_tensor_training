import torch
import torch.nn.functional as F
from src.regression import OneWordTensorRegression, TwoWordTensorRegression, CPTensorRegression

from sentence_transformers import SentenceTransformer


class ModelBank(object):
    def __init__(self, model_locations: str):
        self.reference_caches: dict[str, list] = dict()
        self.model_caches: dict[tuple[str, str], torch.nn.Module] = dict()
        self.BERT_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        self.model_locations = model_locations

    def load_reference(self, lang_type: str, file_path: str):
        with open(file_path, "r") as f:
            self.reference_caches[lang_type] = f.read().splitlines()
        return self.reference_caches[lang_type]

    def retrieve_BERT(self, text: str):
        """
        Retrieves the BERT embedding for the given text.
        """
        word_embedding = self.BERT_model.encode(text, convert_to_tensor=True)
        word_embedding = word_embedding.cpu().numpy().reshape(1, -1)

        return torch.from_numpy(word_embedding)

    def ann(self, target: str, lang_type: str, file_path: str = None, context = ""):
        
        candidates: list[str] = []
        print(lang_type in self.reference_caches)
        if lang_type in self.reference_caches:
            candidates = self.reference_caches[lang_type]
        else:
            if file_path:
                candidates = self.load_reference(lang_type, file_path)
            else:
                raise ValueError(f"Reference list for {lang_type} not found. Please load it first.")
        
        #evaluation:
        max_score = float('-inf')
        best_candidate = None
        target_embedding = self.retrieve_BERT(context + target)

        for candidate in candidates:
            candidate_embedding = self.retrieve_BERT(context + candidate)

            score = F.cosine_similarity(target_embedding, candidate_embedding, dim=1).item()
            
            print(f"Comparing '{target}' with '{candidate}': score = {score}")
            
            if score > max_score:
                max_score = score
                best_candidate = candidate
                if score >= 0.99:
                    break
        
        print(f"Best candidate: '{best_candidate}' with score {max_score}")

        return best_candidate, max_score if best_candidate else "No suitable candidate found"

    def __load_model(self, directory: str, model_name: tuple[str, str], n: int) -> torch.nn.Module:
        """
        load regression model from the given path.
        """
        model_path = f"{self.model_locations}/{directory}/{model_name[0]}"
        
        model = CPTensorRegression([384 for _ in range(n)], 384, 100)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        return model

    def load_ann(self, ID: tuple[str, str], n: int, file_path: str = None):
        """
        load ANN model from the given path.
        """
        if ID in self.model_caches:
            model = self.model_caches[ID]
        else:
            nearest_name = self.ann(ID[0], ID[1], file_path)
            model = self.__load_model(ID[1], nearest_name, n = n)
            self.model_caches[ID] = model

        return model
    


if __name__ == "__main__":
    cache = ModelBank("/mnt/ssd/user-workspaces/aidan-svc/CCD_tensor_training")
    # with open("src/DisCoBERT/ref/tverb.txt", "r") as f:
    #     adv_adj = f.read().splitlines()
    
    # target = "big"


    model = cache.load_ann(("Leafy", "adj_model"), file_path="src/DisCoBERT/ref/adj_model.txt")

    word = cache.retrieve_BERT("greens")

    comparison = cache.retrieve_BERT("leafy greens")

    print("cosine similarity:", F.cosine_similarity(model(word), comparison, dim=1).item())



    #print(model(word))

    

    #print(cache.reference_caches)