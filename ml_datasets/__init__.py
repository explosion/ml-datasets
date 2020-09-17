from ._registry import register_loader as loaders
from .loaders.imdb import imdb
from .loaders.mnist import mnist
from .loaders.quora import quora_questions
from .loaders.reuters import reuters
from .loaders.snli import snli
from .loaders.stack_exchange import stack_exchange
from .loaders.universal_dependencies import ud_ancora_pos_tags, ud_ewtb_pos_tags
from .loaders.dbpedia import dbpedia
from .loaders.cmu import cmu
