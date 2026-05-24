import tiktoken
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer



# Del 01-----------------------------------------------------------


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt).ids
        
        #Ventana deslizante
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

#El DataLoader gestiona cómo se entregan los datos al modelo durante el entrenamiento.
#Como el max_length es 256 (contexto) y el stride (salto) es 128, hay un solapamiento del 50%

from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import DataLoader

def crea_dataloader_v1(txt, tokenizer, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    
    tokenizer = tokenizer

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader



# Del 02-----------------------------------------------------------


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out debe ser divisible por num_heads"

        #Aquí calculamos las dimensiones de query, key, y value
        #Calculamos las dimensiones para todas las cabezas juntas y luego hacemos que cada una tenga la misma         salida
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        #Capa linear para combinar los outputs de cada cabeza
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  #Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        #Hacemos el split añadiendo las dimensiones de numero de cabezas y dimension de cabeza
        #Como d_out = num_heads * head_dim se splitea con view
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # para cada batch hay tokens, para cada token hay cabezas, para cada cabeza hay x valores
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        #Transpongo para posteriormente operar con (...,num_tokens,head_dim)
        #(b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        #Calculo la atención propia enmascarada
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        #Extrae por filas y columnas hasta el número de tokens que tenemos de la máscara inicial y la pasa a          bool siendo los 1s True
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        #Llena los valores True de la máscara con -infinito para aplicar softmax y que sea 0
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        #Normaliza los scores de atención para calcular los pesos de atención
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        #Volvemos a (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        #Se combinan las cabezas, donde self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        #el modelo unifica y refina lo que cada cabezal ha aprendido
        context_vec = self.out_proj(context_vec)  

        return context_vec


# Del 03-----------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        #self.eps es para que el número nunca sea dividido entre 0
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        #Durante el aprendizaje el modelo puede hacer undo sobre x-mean sumando self.shit cuando aprenda que es lo más óptimo.
        #Ajustará de ese modo la normalización en función de su entrenamiento, por eso al principio son 0s, porque el modelo irá desarrollando
        #conforme aumenten las épocas y al principio no regulará.
        #Lo mismo pasará con self.scale,al principio multiplicará los pesos normalizados por 1 (se quedará igual) pero conforme vaya aprendiendo
        #ajustará la división / torch.sqrt(var + self.eps) con la multiplicación de self.scale.
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return 0.5 * x * (1+torch.tanh(
            torch.sqrt(torch.tensor(2.0/torch.pi)) *
            (x + 0.044715 * torch.pow(x,3))
        ))


class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers = nn.Sequential(
            #El hiperparámetro 4 es 4 debido a que simulamos la arquitectura GPT pero podría ser cualquiera
            #ya que al final nos quedará la dimensión de los embed
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
            return self.layers(x)


class TransformerBlock(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        #Primera fase
        shortcut = x
        x = self.norm1(x)
        x = self.att(x) #[batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut #Sumamos los logits al input anterior
    
        #Segunda fase
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut #Sumamos los logits al input anterior
    
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model,idx,max_new_tokens,context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:,-context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        probas = torch.softmax(logits,dim=-1)

        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)
    return idx