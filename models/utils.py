def split_first_n_sentences(text: str, n: int) -> str:
    sentences = re.findall(r"[^.!?]*[.!?]", text, flags=re.UNICODE)
    if n <= 0 or n >= len(sentences):
        return text.strip()
    return "".join(sentences[:n]).strip()

def generate_with_scores(model, prompt_ids, max_new_tokens=200, **gen_kwargs):
    out = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        **gen_kwargs
    )
    return out.sequences, out.scores

def compute_logprobs(scores: List[torch.Tensor], gen_ids: torch.Tensor):
    lps = []
    for logit_step, tok_id in zip(scores, gen_ids):
        lp = F.log_softmax(logit_step[0], dim=-1)[tok_id].item()
        lps.append(lp)
    return lps

def compute_saliency_manual(model, full_ids, prompt_len: int, focus_len: int, device: str,
                            drop_punct: bool=False, tokenizer=None) -> torch.Tensor:
    """
    Salienza grad-based: |∂L/∂embedding| somma.
    `drop_punct`: se True rimuove token di sola punteggiatura dal target.
    """
    model.zero_grad(set_to_none=True)

    focus_end = prompt_len + focus_len
    seq_ids = full_ids[:, :focus_end]

    embeds = model.get_input_embeddings()(seq_ids).detach()
    embeds.requires_grad_()

    attn_mask = torch.ones_like(seq_ids, dtype=torch.long, device=device)
    logits = model(inputs_embeds=embeds, attention_mask=attn_mask).logits

    shift_logits = logits[:, :-1, :]
    shift_labels = seq_ids[:, 1:]

    start = prompt_len - 1
    end   = prompt_len + focus_len - 1

    sel_logits = shift_logits[0, start:end, :]
    sel_labels = shift_labels[0, start:end]

    log_probs = F.log_softmax(sel_logits, dim=-1)
    target_lp = log_probs[torch.arange(focus_len), sel_labels]

    if drop_punct and tokenizer is not None:
        toks = tokenizer.convert_ids_to_tokens(sel_labels.cpu().tolist())
        mask = torch.tensor([any(ch.isalpha() for ch in t) for t in toks],
                            dtype=torch.bool, device=sel_labels.device)
        target_lp = target_lp[mask]

    loss = -target_lp.mean()
    loss.backward()

    grads = embeds.grad[0, :prompt_len, :]
    sal_scores = grads.abs().sum(dim=-1).detach()
    return sal_scores

def plot_saliency(tokens, scores, title=""):
    plt.figure(figsize=(max(8, len(tokens)*0.4), 4))
    plt.bar(range(len(tokens)), scores.cpu().numpy())
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
    plt.ylabel("Saliency (|∂L/∂e| sum)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# --- Word-level aggregation (BPE -> parole) ---
PREFIXES = ("Ġ", "▁")  # GPT-2 usa Ġ, SentencePiece (LLaMA/Mistral) usa ▁

def strip_prefix(tok: str):
    for p in PREFIXES:
        if tok.startswith(p):
            return tok[len(p):]
    return tok

def aggregate_bpe(tokens, scores):
    """
    tokens: lista token BPE del prompt
    scores: tensor/array salienze corrispondenti
    ritorna: DataFrame con parola aggregata e somma salienze
    """
    import pandas as pd
    words, agg_scores = [], []
    cur_word, cur_score = "", 0.0

    for tok, s in zip(tokens, scores.cpu().tolist()):
        clean = strip_prefix(tok)

        # nuovo inizio parola se token ha prefisso o siamo a inizio
        new_boundary = any(tok.startswith(p) for p in PREFIXES) or cur_word == ""

        # punteggiatura isolata -> nuova parola
        if len(clean) == 1 and not clean.isalnum():
            new_boundary = True

        if new_boundary and cur_word != "":
            words.append(cur_word)
            agg_scores.append(cur_score)
            cur_word, cur_score = clean, s
        else:
            cur_word += clean
            cur_score += s

    if cur_word != "":
        words.append(cur_word)
        agg_scores.append(cur_score)

    return pd.DataFrame({"word": words, "score": agg_scores})

def plot_word_saliency(df_words, top_n=20, title="Top parole per salienza"):
    import matplotlib.pyplot as plt
    df_top = df_words.sort_values("score", ascending=False).head(top_n)
    plt.figure(figsize=(10,4))
    plt.bar(range(len(df_top)), df_top["score"])
    plt.xticks(range(len(df_top)), df_top["word"], rotation=45, ha="right")
    plt.ylabel("Saliency (aggregated)")
    plt.title(title)
    plt.tight_layout()
    plt.show()