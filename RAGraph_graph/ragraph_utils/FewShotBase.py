import torch

from .TaskDecoder import TaskDecoder
from .SimilarityFunctions import SimilarityFunctions

class FewShotBase:
    def __init__(self, dataset_name: str, num_classes: int, pretrain_model):
        
        self.fewshot_adj: torch.Tensor = torch.load(f"data/fewshot_{dataset_name}_graph/testset/adj.pt").cuda()
        self.fewshot_feature: torch.Tensor = torch.load(f"data/fewshot_{dataset_name}_graph/testset/feature.pt").cuda()
        self.fewshot_label: torch.Tensor = torch.load(f"data/fewshot_{dataset_name}_graph/testset/labels.pt").cuda()
        self.fewshot_graph_len = torch.load(f"data/fewshot_{dataset_name}_graph/testset/graph_len.pt").cuda()

        self.fewshot_embeddings: torch.Tensor = pretrain_model.inference(self.fewshot_feature, self.fewshot_adj).squeeze()
        self.fewshot_one_hot_label = torch.nn.functional.one_hot(self.fewshot_label.long(), num_classes=num_classes).float().cuda()

    def __call__(self, search_embeddings: torch.Tensor, decoder: TaskDecoder) -> torch.Tensor:
        search_embeddings_decoded = decoder(search_embeddings)
        fewshot_embeddings_decoded = decoder(self.fewshot_embeddings)
        
        # (batch_size, fewshot_num)
        similarity = SimilarityFunctions.calculate_cosine_similarity(search_embeddings_decoded, fewshot_embeddings_decoded)
        
        # (batch_size, num_classes) = (batch_size, fewshot_num) * (fewshot_num, num_classes)
        predict_label = torch.matmul(similarity, self.fewshot_one_hot_label)

        return predict_label