import torch
from torch import tensor as T
import pickle
import argparse

from indexers import DenseFlatIndexer
from encoder import KobertBiEncoder
from dpr_data import KorQuadDataset, KorQuadSampler, korquad_collator
from utils import get_passage_file


class KorDPRRetriever:
    def __init__(self, model, valid_dataset, index, val_batch_size: int = 64):
        self.model = model
        self.tokenizer = valid_dataset.tokenizer
        self.val_batch_size = val_batch_size
        self.valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset.dataset,
            batch_sampler=KorQuadSampler(
                valid_dataset.dataset, batch_size=val_batch_size, drop_last=False
            ),
            collate_fn=lambda x: korquad_collator(
                x, padding_value=valid_dataset.pad_token_id
            ),
            num_workers=4,
        )
        self.index = index

    # def val_top_k_acc(self, k:int=100):
    #     '''validation set에서 top k 정확도를 계산합니다.'''
    #     pass

    # def _top_k_acc(self, loader, k: int=100):
    #     '''주어진 loader에 대해 top-k retrieval 정확도를 계산합니다.'''
    #     pass

    def retrieve(self, query: str, k: int = 100):
        """주어진 쿼리에 대해 가장 유사도가 높은 passage를 반환합니다."""
        tok = self.tokenizer.batch_encode_plus([query])
        with torch.no_grad():
            out = self.model(T(tok["input_ids"]), T(tok["attention_mask"]), "query")
        result = self.index.search_knn(query_vectors=out.cpu().numpy(), top_docs=k)

        # 원문 가져오기
        passages = []
        for idx, sim in zip(*result[0]):
            path = get_passage_file([idx])
            if not path:
                print(f"No single passage path for {idx}")
                continue
            with open(path, "rb") as f:
                passage_dict = pickle.load(f)
            print(f"passage : {passage_dict[idx]}, sim : {sim}")
            passages.append((passage_dict[idx], sim))
        return passages


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", type=str, required=True)
    parser.add_argument("--k", "-k", type=str, required=True)
    args = parser.parse_args()

    model = KobertBiEncoder()
    model.load("checkpoint/2050iter_model.pt")
    valid_dataset = KorQuadDataset("dataset/KorQuAD_v1.0_dev.json")
    index = DenseFlatIndexer()
    index.deserialize(path="2050iter_flat")
    retriever = KorDPRRetriever(model=model, valid_dataset=valid_dataset, index=index)
    retriever.retrieve(query = args.query, k = args.k)
    # retriever.retrieve(query="중국의 천안문 사태가 일어난 연도는?", k=20)
