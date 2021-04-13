# -*- coding: utf-8 -*-
import torch
torch.cuda.synchronize()
start = time.time()
a = torch.rand(4, 3)
# b = torch.rand(1)
# c = a+b
# print("a", a, "b", b, "c", c)
print(a.mean())
end = time.time()