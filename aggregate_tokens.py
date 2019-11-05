
# N:  batch_size
# R:  number of each example's relation facts
# MP: number of mention pairs of each relation fact. For example, If the head and tail entity has 3 and 2 mentions respectively, then `MP=6` in this case.
# E:  embedding dimension, e.g., 128

# sequence_repr:          representation of the document (i.e., abstract in your paper)
# entity_span_indices:    indices of all mentions in abstract (suppose that all mentions are single-token mention, so each index points to the mention)
# head_tail_comb_indices: (N, R, MP, 2) this holds the head/tail mentions' indices. [..., 0] for head mentions, [..., 1] for tail mentions.
# Supposed that the indices of head mentions are 1, 8, 9; and the tail mentions are 5, 6, then the `head_tail_comb_indices` for this realtion fact is:
# ```
# 	head: head_tail_comb_indices[bsz_index, rel_idx, :, 0] = [1, 1, 8, 8, 9, 9]
# 	tail: head_tail_comb_indices[bsz_index, rel_idx, :, 1] = [5, 6, 5, 6, 5, 6]
# ```
# Shape: (N, num_mentions, embedding_dim), this module obtains all mention's representation
span_embeddings = self.entity_spans_embeddings(sequence_repr, entity_span_indices)

# project each contextually encoded mention to head or tail, i.e., first two formulas in Sec. 2.3 of your paper
# Shape: (N, num_mentions, embedding_dim)
head_span_embeddings = self.mlp2head(span_embeddings)
# Shape: (N, num_mentions, embedding_dim)
tail_span_embeddings = self.mlp2tail(span_embeddings)

N, R, MP, _ = head_tail_comb_indices.size()
# (N, R*MP, 2), reshpae to (N, R*MP, 2)
head_tail_comb_indices = head_tail_comb_indices.view(N, R*MP, 2)

# Select representations of head/tail mentions from `span_embeddings`
# Shape: (N, R*MP, E)
head_embeddings = util.batched_index_select(head_span_embeddings, head_tail_comb_indices[:, :, 0])
# Shape: (N, R*MP, E)
tail_embeddings = util.batched_index_select(tail_span_embeddings, head_tail_comb_indices[:, :, 1])

# apply dropout
# Shape: (N, R*MP, E)
head_embeddings = self.head_drop(head_embeddings)
# Shape: (N, R*MP, E)
tail_embeddings = self.tail_drop(tail_embeddings)

# apply bilinear to each mention pair
# (N, R*MP, num_relation)
logits = self.bili(head_embeddings, tail_embeddings)

# (N, R, MP, num_relation), reshape to (N, R, MP, num_relation)
logits = logits.view(N, R, MP, self.num_relation)

# Mask out padding. mask out some padding relations or padding mention pairs.
logits = logits - (1-comb_rel_embeddings_indices_mask.unsqueeze(-1).float())*1e10

# (N, R, num_relation), reduction on the mention pairs dim. i.e., aggregate the relation scores described in Sec 2.4 of your paper
logits = torch.logsumexp(logits, dim=2)

