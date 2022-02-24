import torch
from torch import tensor as T
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel
import transformers

transformers.logging.set_verbosity_error()  # 토크나이저 초기화 관련 warning suppress
from tqdm import tqdm
import os
import logging
from typing import List, Tuple

from chunk_data import DataChunk
from encoder import KobertBiEncoder
import indexers
from utils import get_wiki_filepath, wiki_worker_init

# logger basic config
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/log.log",
    level=logging.DEBUG,
    format="[%(asctime)s | %(funcName)s @ %(pathname)s] %(message)s",
)
logger = logging.getLogger()


def wiki_collator(batch: List, padding_value: int) -> Tuple[torch.Tensor]:
    """passage를 batch로 반환합니다."""
    batch_p = pad_sequence(
        [T(e) for e in batch], batch_first=True, padding_value=padding_value
    )
    batch_p_attn_mask = (batch_p != padding_value).long()
    return (batch_p, batch_p_attn_mask)


class WikiArticleStream(torch.utils.data.IterableDataset):
    """
    Indexing을 위해 random access가 필요하지 않고 large corpus를 다루기 위해 stream dataset을 사용합니다.
    """

    def __init__(self, wiki_path, chunker):
        # self.chunk_size = chunk_size
        super(WikiArticleStream, self).__init__()
        self.chunker = chunker
        self.pad_token_id = self.chunker.tokenizer.get_vocab()["[PAD]"]
        self.wiki_path = wiki_path
        self.max_length = 168  # maximum length for kowiki passage
        # self.start = 0
        # self.end = len(self.passages)

    def __iter__(self):
        # max_length가 되도록 padding 수행

        _, passages = self.chunker.chunk(self.wiki_path)
        logger.debug(f"chunked file {self.wiki_path}")
        for passage in passages:
            # if len(passage) > self.max_length:
            #     continue  # 지정된 max_length보다 긴 passage의 경우 pass
            # padded_passage = T(
            #     passage
            #     + [self.pad_token_id for _ in range(self.max_length - len(passage))]
            # )
            yield passage


class IndexRunner:
    """코퍼스에 대한 인덱싱을 수행하는 메인클래스입니다. passage encoder와 data loader, FAISS indexer로 구성되어 있습니다."""

    def __init__(
        self,
        data_dir: str,
        model_ckpt_path: str,
        indexer_type: str = "DenseFlatIndexer",
        chunk_size: int = 100,
        batch_size: int = 64,
        buffer_size: int = 50000,
        index_output: str = "",
        device: str = "cuda:0",
    ):
        """
        data_dir : 인덱싱할 한국어 wiki 데이터가 들어있는 디렉토리입니다. 하위에 AA, AB와 같은 디렉토리가 있습니다.
        indexer_type : 사용할 FAISS indexer 종류로 DPR 리포에 있는 대로 Flat, HNSWFlat, HNSWSQ 세 종류 중에 사용할 수 있습니다.
        chunk_size : indexing과 searching의 단위가 되는 passage의 길이입니다. DPR 논문에서는 100개 token 길이 + title로 하나의 passage를 정의하였습니다.
        """
        if "=" in data_dir:
            self.data_dir, self.to_this_page = data_dir.split("=")
            self.to_this_page = int(self.to_this_page)
            self.wiki_files = get_wiki_filepath(self.data_dir)
        else:
            self.data_dir = data_dir
            self.wiki_files = get_wiki_filepath(self.data_dir)
            self.to_this_page = len(self.wiki_files)

        self.device = torch.device(device)
        self.encoder = KobertBiEncoder().to(self.device)
        self.encoder.load(model_ckpt_path)  # loading model
        self.indexer = getattr(indexers, indexer_type)()
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.loader = self.get_loader(
            self.wiki_files[: self.to_this_page],
            chunk_size,
            batch_size,
            worker_init_fn=None,
        )
        self.indexer.init_index(self.encoder.emb_sz)
        self.index_output = index_output if index_output else indexer_type

    @staticmethod
    def get_loader(wiki_files, chunk_size, batch_size, worker_init_fn=None):
        chunker = DataChunk(chunk_size=chunk_size)
        ds = torch.utils.data.ChainDataset(
            tuple(WikiArticleStream(path, chunker) for path in wiki_files)
        )
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            collate_fn=lambda x: wiki_collator(
                x, padding_value=chunker.tokenizer.get_vocab()["[PAD]"]
            ),
            num_workers=1,
            worker_init_fn=worker_init_fn,
        )  # TODO : chain dataset에서 worker 1 초과인 경우 확인하기
        return loader

    def run(self):
        _to_index = []
        cur = 0
        for batch in tqdm(self.loader, desc="indexing"):
            p, p_mask = batch
            p, p_mask = p.to(self.device), p_mask.to(self.device)
            with torch.no_grad():
                p_emb = self.encoder(p, p_mask, "passage")
            _to_index += [(cur + i, _emb) for i, _emb in enumerate(p_emb.cpu().numpy())]
            cur += p_emb.size(0)
            if len(_to_index) > self.buffer_size - self.batch_size:
                logger.debug(f"perform indexing... {len(_to_index)} added")
                self.indexer.index_data(_to_index)
                _to_index = []
        if _to_index:
            logger.debug(f"perform indexing... {len(_to_index)} added")
            self.indexer.index_data(_to_index)
            _to_index = []
        os.makedirs(self.index_output, exist_ok=True)
        self.indexer.serialize(self.index_output)


if __name__ == "__main__":
    IndexRunner(
        data_dir="text",
        model_ckpt_path="checkpoint/2050iter_model.pt",
        index_output="2050iter_flat",
    ).run()
    # test_loader()
