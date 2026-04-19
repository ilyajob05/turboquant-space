"""Encode PyTorch embeddings into TurboQuant compact codes."""
import numpy as np
import torch

from turboquant import TurboQuantSpace


def main() -> None:
    torch.manual_seed(0)
    n, dim = 1000, 768
    x = torch.randn(n, dim)
    x = torch.nn.functional.normalize(x, dim=1)

    tq = TurboQuantSpace(dim=dim, bits_per_coord=8)

    x_np = x.detach().cpu().numpy().astype(np.float32, copy=False)
    x_np = np.ascontiguousarray(x_np)
    codes = tq.encode_batch(x_np)

    raw_bytes = x.numel() * 4
    comp_bytes = codes.nbytes
    print(f"raw   : {raw_bytes / 1e6:.2f} MB")
    print(f"codes : {comp_bytes / 1e6:.2f} MB  ({raw_bytes / comp_bytes:.1f}x)")
    print(f"code_size_bytes = {tq.code_size_bytes()}")

    q = x_np[0]
    d = tq.distance_1_to_n(q, codes)
    print("top-5 nearest:", np.argsort(d)[:5])


if __name__ == "__main__":
    main()
