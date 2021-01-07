import  torch
import  torch.nn.functional as f
# A = torch.randint(2,size=(2,5,5)).float()
def Doubly_normalization(adj_old):
    E_ij = torch.nn.functional.normalize(adj_old,p=1,dim=2)
    adj_new= torch.matmul(E_ij, torch.transpose(torch.div(E_ij, torch.sum(E_ij, dim=1).squeeze()), dim0=1, dim1=2))
    return adj_new
A = torch.tensor([[
    [1., 1., 0., 0., 0., 0.],
    [0., 1., 1., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0.],
    [0., 0., 0., 1., 1., 0.],
    [0., 0., 1., 0., 1., 0.],
    [0., 0., 1., 0., 0., 1.]],
    [[1., 1., 0., 0., 0.,],
    [0., 1., 1., 0., 0., ],
    [0., 0., 1., 0., 0., ],
    [0., 0., 0., 1., 1., ],
    [0., 0., 1., 0., 1.,],]
])

E_ij = f.normalize(A,p=1,dim=2)

C = torch.sum(E_ij,dim=1).unsqueeze(dim=2)
print(C.shape)
seq = E_ij.shape[1]
# D = C.repeat([1,1,seq])


# print(torch.div(E_ij,torch.sum(E_ij,dim=1).repeat([1,1,seq])))
Evk = torch.matmul(E_ij,torch.transpose(torch.div(E_ij,torch.sum(E_ij,dim=1).unsqueeze(1).repeat([1,seq,1])),dim0=1,dim1=2))
print(Evk)
print(Evk.shape)
# print(Evk)
