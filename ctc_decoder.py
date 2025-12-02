import torch
import k2
from collections import defaultdict
from typing import Optional, Union, TypeVar, Literal, NamedTuple, Sequence, overload

T = TypeVar('T', int, str)
Recursive = Union[T, Sequence['Recursive[T]']]
RecursiveTuple = Union[T, tuple['RecursiveTuple[T]']]
Symbols = Sequence[Recursive[int]]
StringSymbols = Sequence[Recursive[str]]
AnySymbols = Union[Symbols, StringSymbols]

class Probabilities(NamedTuple):
    probabilities: torch.Tensor
    classifications: AnySymbols

@overload
def decode_probabilities(symbols: Symbols, symbol_of_interest: Optional[int], logits: torch.Tensor, vocab: None, pad_token_id: int, as_strings: Literal[False]) -> Probabilities:
    pass
@overload
def decode_probabilities(symbols: StringSymbols, symbol_of_interest: Optional[int], logits: torch.Tensor, vocab: dict[str, Recursive[int]], pad_token_id: int, as_strings: Literal[True]) -> Probabilities:
    pass
def decode_probabilities(symbols: AnySymbols, symbol_of_interest: Optional[int], logits: torch.Tensor, vocab: Optional[dict[str, Recursive[int]]] = None, pad_token_id: int = 0, as_strings: bool = False) -> Probabilities:
    """
    Given some possible decodings, returns the relative probabilities of each.

    Args:
        symbols: A list of possible decoding paths to consider. A nested list
            as in ['b', ['a', 'u'], 't'] refers to possibilities ('bat' and 'but'),
            while a doubly nested list as in ['b', ['a', ['o', 'u']], 't'] refers
            back to symbols ('bat' and 'bout'), recursively.
        symbol_of_interest: The index in `symbols` containing the variant paths
            to compare the probabilities of. If `symbols = ['b', ['a', 'u'], 't']`
            and `symbol_of_interest = 1`, the output will contain the respective
            probabilities of the decodings 'bat' and 'but'. Must refer to a list,
            which may itself contain lists and lists of lists. Only top-level
            variant paths are considered. If none, considers all possible decodings.
        logits: A torch.Tensor of shape `(N, T, C)`, where `N` is the batch size,
            `T` is the length the longest item in the batch, and `C` is the
            vocabulary size.
        vocab: A dictionary of the vocabulary. Required if `as_strings` is true,
            otherwise ignored.
        pad_token_id: The padding token. If not 0, the logits must be reorganized
            to be k2-compliant.
        as_strings: Whether the input symbols are strings. If they are, the
            vocabulary is used to convert them to integers.
    """

    if as_strings:
        if not vocab:
            raise ValueError('vocab required when as_strings True!')
        symbols = _str_to_symbols(symbols, vocab)
    
    if symbol_of_interest is not None and (not isinstance(symbols[symbol_of_interest], list) or not len(symbols[symbol_of_interest]) > 1):
        raise ValueError('Symbol of interest must have multiple options!')

    if not pad_token_id == 0:
        symbols = _zero_pad_token_id(symbols, pad_token_id)
        logits = _zero_pad_token_id_logits(logits, pad_token_id)
    
    decodings = _get_decodings(symbols, logits)
    symbols_lists = get_symbols_lists(symbols, symbol_of_interest)
    scores = _get_scores(symbols_lists, decodings)
    probabilities = scores.softmax(1)
    classifications = [symbols_list[symbol_of_interest] if symbol_of_interest is not None else symbols_list for symbols_list in symbols_lists]
    
    if not pad_token_id == 0:
        classifications = _unzero_pad_token_id(classifications, pad_token_id)

    if as_strings:
        classifications = _symbols_to_str(classifications, vocab)

    return Probabilities(probabilities, classifications)

def get_symbols_lists(symbols: AnySymbols, symbol_of_interest: Optional[int]) -> list[AnySymbols]:
    """
    Convenience function to easily distinguish paths of interest.

    For example, if you want to allow for decodings that begin with any consonant
    `['p', 't']` and end with `'b'`, but you are specifically interested in
    whether the medial vowel is either `'i'` or `'u'`., you would enter
    `symbols = [['p', 't'], ['i', 'u'], 'b']` and `symbol_of_interest = 1`.
    This would return the list of symbols lists of interest
    `[[['p', 't'], 'i', 'b'], [['p', 't'], 'u', 'b']]` as required by, e.g.,
    `_get_tot_scores`. This way, you don't have to separately write these out.

    The function is agnostic to whether these are symbols or characters,
    and to whether the symbol of interest is a symbol or a list of symbols.
    This means that something like `symbols = ['t', ['u', ['j', 'u']], 'n']`
    with `symbol_of_interest = 1`, comparing the decodings 'tun' and 'tjun',
    is permitted.
    """

    if symbol_of_interest is not None and (symbol_of_interest < 0 or symbol_of_interest > len(symbols)):
        raise ValueError(f'Symbol of interest {symbol_of_interest} not in range!')
    
    symbols_of_interest = symbols[symbol_of_interest] if symbol_of_interest is not None else symbols
    if not isinstance(symbols_of_interest, list):
        raise ValueError(f'Symbol of interest {symbol_of_interest} not a list!')

    if symbol_of_interest is not None:
        symbols_lists = [symbols[:symbol_of_interest] + symbol + symbols[symbol_of_interest + 1:] for symbol in _flatten_symbol(symbols_of_interest)]
    else:
        symbols_lists = _flatten_symbol(symbols_of_interest)
    return symbols_lists

class Nodes(dict):
    def __missing__(self, key): # newest node should have highest number
        self[key] = len(self)
        return self[key]

def _flatten_symbol(symbols: Symbols) -> Symbols:
    """
    Converts a list of symbols like [[A, B], [C, D]] to the equivalent single-symbol list [[[A, C], [A, D], [B, C], [B, D]]].
    """

    flattened_symbols = [[]]

    for symbol in symbols:
        if isinstance(symbol, list):
            flattened_symbols = [flattened_symbol + [i] for i in symbol for flattened_symbol in flattened_symbols]
        else:
            flattened_symbols = [flattened_symbols + [symbol]]

    return flattened_symbols

def _str_to_symbols(string: StringSymbols, vocab: dict[str, Recursive[int]]) -> Symbols:
    """Translates strings to symbols using a vocabulary. Multiple paths are permitted."""

    return [vocab[char] if isinstance(char, str) else _str_to_symbols(char, vocab) for char in string]

def _symbols_to_str(symbols: Symbols, vocab: dict[str, Recursive[int]]) -> StringSymbols:
    """Translates symbols to strings using a vocabulary. Multiple paths are permitted."""

    reversed_vocab = {value: key for key, value in vocab.items() if isinstance(value, int)}
    return [reversed_vocab[symbol] if isinstance(symbol, int) else _symbols_to_str(symbol, vocab) for symbol in symbols]

def _get_paths(symbols: Symbols, prev_symbol = 0, nodes: Optional[Nodes] = None, edges: Optional[defaultdict] = None, position = 1) -> tuple[Nodes, defaultdict]:
    """
    Recursive paths from previous symbol to remaining symbols as a list of nodes and edges.
    
    A symbol can be one character or a list of possible characters at a given position.
    A possible character might be a list of symbols.

    For example,
        symbols = [1, 2, 3]
    would result in the single possible decoding [1, 2, 3], while
        symbols = [1, [3, [2, 3]]]
    would result in the possible decodings [1, 3] and [1, 2, 3]. On the other hand,
        symbols = [1, 3, [2, 3]]
    would result in the possible decodings [1, 3, 2] and [1, 3, 3]. Finally,
        symbols = [1, [3, [2, [3, 4]]]]
    would result in the possible decodings [1, 3], [1, 2, 3], and [1, 2, 4].
    """

    if not nodes:
        nodes = Nodes() # key: (symbol, position) to prevent
    if not edges:
        edges = defaultdict(lambda: set())
    prev_node = nodes[(prev_symbol, position - 1)] # create initial node if doesn't exist

    if not symbols: # last node
        edges[prev_node].add(-1)
        return nodes, edges
    
    if isinstance(symbols[0], int): # create edge from previous node to current node(s)
        edges[prev_node].add(nodes[(symbols[0], position)])
        return _get_paths(symbols[1:], prev_symbol = symbols[0], nodes = nodes, edges = edges, position = position + 1)
    else:
        if not symbols[0]:
            edges[prev_node].add(-1)
            return nodes, edges
        for symbol in symbols[0]: # re-run on each symbol
            nodes, edges = _get_paths([*symbol, *symbols[1:]] if isinstance(symbol, list) else [symbol, *symbols[1:]], prev_symbol = prev_symbol, nodes = nodes, edges = edges, position = position)
        return nodes, edges

def _paths_to_fst(paths: tuple[Nodes, defaultdict]) -> k2.Fsa:
    """Takes the return value of get_paths and generates a k2-compliant FST."""

    fst_str = set()
    nodes, edges = paths
    last_node = len(nodes)
    node_to_symbol = {node: key[0] for key, node in nodes.items()}
    node_to_symbol[last_node] = -1 # -1 goes to last node

    for from_node, to_nodes in edges.items():
        for to_node in to_nodes:
            if to_node == -1: # -1 goes to last node
                to_node = len(nodes)
            fst_str.add(f'{from_node} {to_node} {node_to_symbol[to_node]} {node_to_symbol[to_node]} 0')
    
    fst_str = '\n'.join(sorted(fst_str, key = lambda x: int(x.split()[0])) + [str(len(nodes))])
    return k2.arc_sort(k2.Fsa.from_str(fst_str, acceptor = False)) # type: ignore # ret_arc_map is False

def _get_decodings(symbols: Symbols, logits: torch.Tensor) -> k2.Fsa:
    """Given a list of symbols and logits, returns the lattice of CTC decodings."""

    a_fst = k2.ctc_topo(logits.shape[2]) # vocabulary size is size of dimension 2
    b_fst = _paths_to_fst(_get_paths(symbols))

    ab_fst = k2.connect(k2.compose(a_fst, b_fst)) # graph of all possible decodings
    decodings = k2.get_lattice(logits, torch.full_like(logits[:, 0, 0], logits.shape[1]), ab_fst, search_beam = 1000, output_beam = 1000, min_active_states = 0)

    return decodings

def _get_scores(symbols_lists: list[Symbols], decodings: k2.Fsa) -> torch.Tensor:
    """
    Gives scores for different possible decodings from a list of mutually exclusive decodings.

    Returns a tensor of shape (N, C), where N is the batch size and C is the number
    of lists to compare.
    
    Each item in `symbols_lists` should be a full, possible, and unique set of possible decodings
    to compose with the decodings.

    For example, if `decodings` was generated from the symbols `[1, [2, 3]]`, which would
    allow the possible decodings `[1, 2]` and `[1, 3]`, you could pass
    `symbols_lists = [[1, 2], [1, 3]]` which would give you the total scores, respectively,
    for the decodings `[1, 2]` and `[1, 3]`. You will usually want to use `torch.softmax`
    to convert this into a list of probabilities.
    """

    if any(symbols_lists.count(x) > 1 for x in symbols_lists): # non-unique
        raise ValueError(f'Symbols lists {symbols_lists} contains duplicate decodings.')
    scores = torch.empty(decodings.shape[0], len(symbols_lists))

    for i, symbols in enumerate(symbols_lists):
        subset_fst = _paths_to_fst(_get_paths(symbols))
        decodings_subset_fst = k2.connect(k2.compose(decodings, subset_fst))
        if decodings_subset_fst[0].shape[0] == 0: # path invalid
            raise ValueError(f'Symbols {symbols} not found in decodings.')
        with torch.no_grad():
            scores[:, i] = decodings_subset_fst.get_tot_scores(True, True)
    
    return scores

def _zero_pad_token_id_logits(logits: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """k2 requires the padding token to be 0; this method allows non-compliant logits to be made compliant."""
    
    logits = torch.cat([logits[:, :, pad_token_id].unsqueeze(-1), logits[:, :, :pad_token_id], logits[:, :, pad_token_id + 1:]], dim = 2)
    return logits

def _zero_pad_token_id(symbols: Symbols, pad_token_id: int) -> Symbols:
    """Takes a list of symbols and shifts them to the adjusted vocabulary for a non-zero pad_token_id."""

    translated_symbols = []
    for symbol in symbols:
        if isinstance(symbol, int):
            if symbol < pad_token_id:
                translated_symbols.append(symbol + 1)
            elif symbol > pad_token_id:
                translated_symbols.append(symbol)
            else:
                raise ValueError('pad_token_id cannot be in symbols!')
        else:
            translated_symbols.append(_zero_pad_token_id(symbol, pad_token_id))
    
    return translated_symbols

def _unzero_pad_token_id(symbols: Symbols, pad_token_id: int) -> Symbols:
    """Takes a translated list of symbols and unshifts them to their original vocabulary."""

    untranslated_symbols = []
    for symbol in symbols:
        if isinstance(symbol, int):
            if symbol == 0:
                raise ValueError('0 cannot be in translated symbols!')
            if symbol <= pad_token_id:
                untranslated_symbols.append(symbol - 1)
            else:
                untranslated_symbols.append(symbol)
        else:
            untranslated_symbols.append(_unzero_pad_token_id(symbol, pad_token_id))
    
    return untranslated_symbols