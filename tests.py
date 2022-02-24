import torch
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer
from torch import tensor as T
from tqdm import tqdm
import pytest
from encoder import KobertBiEncoder

from utils import get_passage_file
from indexers import DenseFlatIndexer
from dpr_data import KorQuadDataset, KorQuadSampler, korquad_collator
from utils import get_wiki_filepath
from chunk_data import DataChunk
from index_runner import WikiArticleStream, IndexRunner


@pytest.mark.skip()
def test_WikiArticleDatasetAndBatchSize():
    max_length = 168
    batch_size = 10
    chunker = DataChunk(chunk_size=100)
    ds = WikiArticleStream("result/AA/wiki_24", chunker=chunker)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=0)
    for e in loader:
        assert e.size() == torch.Size([batch_size, max_length])
        break


# @pytest.mark.skip()
def test_kobertVanilaIndexSearch():
    import pickle

    query_txt = "이탈리아의 포르탈레그르 현은 몇 개의 자치단체로 이루어져 있는가?"
    model = KobertBiEncoder().to("cuda:0")
    model.load("1700_iter_model.pt")
    tok = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
    inputs = tok.batch_encode_plus([query_txt])
    with torch.no_grad():
        out = model(
            T(inputs["input_ids"]).to("cuda:0"),
            T(inputs["attention_mask"]).to("cuda:0"),
            "query",
        )

    my_indexer = DenseFlatIndexer()
    my_indexer.deserialize("1700iter_flat")
    result = my_indexer.search_knn(query_vectors=out.cpu().numpy(), top_docs=10)

    # 원문 가져오기
    # passage_path = get_passage_file(result[0][0])

    for idx, sim in zip(*result[0]):
        path = get_passage_file([idx])
        if not path:
            print(f"No single passage path for {idx}")
            continue
        with open(path, "rb") as f:
            passage_dict = pickle.load(f)
        print(f"passage : {passage_dict[idx]}, sim : {sim}")

    assert True


@pytest.mark.skip()
def test_makeOnePageIndex():
    IndexRunner(data_dir="result=1", index_output="flat_test").run()


@pytest.mark.skip()
def test_DataLoaderArticleCounts():
    wiki_files = get_wiki_filepath("result")
    print(f"number of wiki files to process : {len(wiki_files)}")
    ld = IndexRunner.get_loader(
        wiki_files[:1], chunk_size=100, batch_size=64, worker_init_fn=None
    )
    total_num = 0
    for batch in tqdm(ld):
        total_num += batch.size(0)
    assert (
        total_num == 2591
    ), f"total passage number should be 2591 but it is {total_num}!"
    # total number of passages : 2910127


@pytest.mark.skip()
def test_korquadSampler():
    ds = KorQuadDataset("dataset/KorQuAD_v1.0_dev.json")
    ds.load()
    sampler = KorQuadSampler(ds.tokenized_tuples, batch_size=10)
    # assert len(list(sampler)) == len(sampler), f"__len__ method is wrong. expected {len(sampler)} but got {len(list(sampler))}"
    samples = list(sampler)[0]
    batch_set = set([ds.data_tuples[i][1] for i in samples])
    assert len(batch_set) == len(
        samples
    ), "answer passages within a batch should be unique!"


@pytest.mark.skip()
def test_dprDataLoader():
    ds = KorQuadDataset(korquad_path="dataset/KorQuAD_v1.0_dev.json")
    loader = torch.utils.data.DataLoader(
        dataset=ds.dataset,
        batch_sampler=KorQuadSampler(ds.dataset, batch_size=16, drop_last=False),
        collate_fn=lambda x: korquad_collator(x, padding_value=ds.pad_token_id),
        num_workers=4,
    )
    # print(len(_dataset.tokenized_tuples))
    torch.manual_seed(123412341235)
    # cnt = 0
    for batch in tqdm(loader):
        # print(len(batch))
        # cnt+=batch[0].size(0)
        # break
        assert all(
            16 == e.size(0) for e in batch
        ), f"batch size is set to 16 but got {[e.size(0) for e in batch]}"
    # print(cnt)


if __name__ == "__main__":
    # test_WikiArticleStreamDatset()
    test_kobertVanilaIndexSearch()
    # test_makeOnePageIndex()
