import numpy as np
import cv2
import torch
# import kornia
import os

def SDR_to_YCbCr(x, dim=1):

    transform_matrix = torch.tensor([
        [0.299,  0.587,  0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ]).to(x.device)
    
    if x.dim() == 4:
        B, C, H, W = x.size()
        x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
        ycbcr = torch.matmul(x_reshaped, transform_matrix.T)
        ycbcr = ycbcr.view(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
    elif x.dim() == 3:
        C, H, W = x.size()
        x_reshaped = x.permute(1, 2, 0).contiguous().view(-1, C)
        ycbcr = torch.matmul(x_reshaped, transform_matrix.T)
        ycbcr = ycbcr.view(H, W, 3).permute(2, 0, 1).contiguous()
    else:
        raise ValueError("输入张量的维度必须为3或4")
    
    return ycbcr

def YCbCr_to_SDR(x, dim=1):

    inverse_transform_matrix = torch.tensor([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ]).to(x.device)
    
    if x.dim() == 4:
        B, C, H, W = x.size()
        x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
        rgb = torch.matmul(x_reshaped, inverse_transform_matrix.T)
        rgb = rgb.view(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
    elif x.dim() == 3:
        C, H, W = x.size()
        x_reshaped = x.permute(1, 2, 0).contiguous().view(-1, C)
        rgb = torch.matmul(x_reshaped, inverse_transform_matrix.T)
        rgb = rgb.view(H, W, 3).permute(2, 0, 1).contiguous()
    else:
        raise ValueError("输入张量的维度必须为3或4")
    
    rgb = torch.clamp(rgb, 0.0, 1.0)  # 假设数据范围是 [0, 1]
    
    return rgb

def ESDR709_to_LSDR2020(ERGB709, dim):
    LRGB = ERGB709 ** 2.4
    LR, LG, LB = torch.split(LRGB, 1, dim=dim)  # hw1
    LR2020 = 0.6274 * LR + 0.3293 * LG + 0.0433 * LB
    LG2020 = 0.0691 * LR + 0.9195 * LG + 0.0114 * LB
    LB2020 = 0.0164 * LR + 0.0880 * LG + 0.8956 * LB
    LRGB2020 = torch.cat([LR2020, LG2020, LB2020], dim=dim)  # hw3
    return LRGB2020 * 100



def LSDR2020_to_ESDR709(LRGB2020, dim=-1):
    M = torch.tensor([
        [0.6274, 0.3293, 0.0433],
        [0.0691, 0.9195, 0.0114],
        [0.0164, 0.0880, 0.8956]
    ], dtype=LRGB2020.dtype, device=LRGB2020.device)
    M_inv = torch.inverse(M)  

    LRGB2020 = LRGB2020 / 100.0

    LR2020, LG2020, LB2020 = torch.split(LRGB2020, 1, dim=dim)  

    LR709_linear = (
        M_inv[0, 0] * LR2020 + M_inv[0, 1] * LG2020 + M_inv[0, 2] * LB2020
    )
    LG709_linear = (
        M_inv[1, 0] * LR2020 + M_inv[1, 1] * LG2020 + M_inv[1, 2] * LB2020
    )
    LB709_linear = (
        M_inv[2, 0] * LR2020 + M_inv[2, 1] * LG2020 + M_inv[2, 2] * LB2020
    )

    LRGB709_linear = torch.cat([LR709_linear, LG709_linear, LB709_linear], dim=dim)

    ERGB709 = torch.clamp(LRGB709_linear, min=1e-10) ** (1.0 / 2.4)
    return ERGB709


def EOTF_PQ_cuda(ERGB):
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    ERGB = torch.clamp(ERGB, min=1e-10, max=1)

    X1 = ERGB ** (1 / m2)
    X2 = X1 - c1
    X2[X2 < 0] = 0

    X3 = c2 - c3 * X1

    X4 = (X2 / X3) ** (1 / m1)
    return X4 * 10000


def EOTF_PQ_cuda_inverse(LRGB):
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875
    RGB_l = LRGB / 10000
    RGB_l = torch.clamp(RGB_l, min=1e-10, max=1)

    X1 = c1 + c2 * RGB_l ** m1
    X2 = 1 + c3 * RGB_l ** m1
    X3 = (X1 / X2) ** m2
    return X3


def HDR_to_ICTCP(ERGB, dim=-1):
    LRGB = EOTF_PQ_cuda(ERGB)  # hw3
    LR, LG, LB = torch.split(LRGB, 1, dim=dim)  # hw1

    L = (1688 * LR + 2146 * LG + 262 * LB) / 4096
    M = (683 * LR + 2951 * LG + 462 * LB) / 4096
    S = (99 * LR + 309 * LG + 3688 * LB) / 4096
    LMS = torch.cat([L, M, S], dim=dim)  # hw3

    ELMS = EOTF_PQ_cuda_inverse(LMS)  # hw3

    EL, EM, ES = torch.split(ELMS, 1, dim=dim)  # hw1
    I = (2048 * EL + 2048 * EM + 0 * ES) / 4096
    T = (6610 * EL - 13613 * EM + 7003 * ES) / 4096
    P = (17933 * EL - 17390 * EM - 543 * ES) / 4096

    ITP = torch.cat([I, T, P], dim=dim)  # hw3
    return ITP


def SDR_to_ICTCP(ERGB, dim=-1):
    LRGB = ESDR709_to_LSDR2020(ERGB, dim=dim)

    LR, LG, LB = torch.split(LRGB, 1, dim=dim)  # hw1
    L = (1688 * LR + 2146 * LG + 262 * LB) / 4096
    M = (683 * LR + 2951 * LG + 462 * LB) / 4096
    S = (99 * LR + 309 * LG + 3688 * LB) / 4096
    LMS = torch.cat([L, M, S], dim=dim)  # hw3

    ELMS = EOTF_PQ_cuda_inverse(LMS)  # hw3

    EL, EM, ES = torch.split(ELMS, 1, dim=dim)  # hw1
    I = (2048 * EL + 2048 * EM + 0 * ES) / 4096
    T = (6610 * EL - 13613 * EM + 7003 * ES) / 4096
    P = (17933 * EL - 17390 * EM - 543 * ES) / 4096

    ITP = torch.cat([I, T, P], dim=dim)  # hw3
    return ITP


def ICTCP_to_HDR(ITP, dim=-1):
    I, T, P = torch.split(ITP, 1, dim=dim)  # hw1
    EL = 1 * I + 0.009 * T + 0.111 * P
    EM = 1 * I - 0.009 * T - 0.111 * P
    ES = 1 * I + 0.560 * T - 0.321 * P
    ELMS = torch.cat([EL, EM, ES], dim=dim)  # hw3

    LMS = EOTF_PQ_cuda(ELMS)
    L, M, S = torch.split(LMS, 1, dim=dim)  # hw1

    X = 2.071 * L - 1.327 * M + 0.207 * S
    Y = 0.365 * L + 0.681 * M - 0.045 * S
    Z = -0.049 * L - 0.05 * M + 1.188 * S

    R = 1.7176 * X - 0.3557 * Y - 0.2534 * Z
    G = -0.6667 * X + 1.6165 * Y + 0.0158 * Z
    B = 0.0176 * X - 0.0428 * Y + 0.9421 * Z

    RGB = torch.cat([R, G, B], dim=dim)  # hw3
    ERGB = EOTF_PQ_cuda_inverse(RGB)

    return ERGB

def ICTCP_to_SDR(ITP, dim=1):
    I, T, P = torch.split(ITP, 1, dim=dim)  
    EL = 1 * I + 0.009 * T + 0.111 * P
    EM = 1 * I - 0.009 * T - 0.111 * P
    ES = 1 * I + 0.560 * T - 0.321 * P
    ELMS = torch.cat([EL, EM, ES], dim=dim) 

    LMS = EOTF_PQ_cuda(ELMS)  
    L, M, S = torch.split(LMS, 1, dim=dim)  


    X = 2.071 * L - 1.327 * M + 0.207 * S
    Y = 0.365 * L + 0.681 * M - 0.045 * S
    Z = -0.049 * L - 0.05 * M + 1.188 * S

    R = 1.7176 * X - 0.3557 * Y - 0.2534 * Z
    G = -0.6667 * X + 1.6165 * Y + 0.0158 * Z
    B = 0.0176 * X - 0.0428 * Y + 0.9421 * Z
    RGB_hdr_linear = torch.cat([R, G, B], dim=dim)  # [B,3,H,W]

    ERGB_sdr = LSDR2020_to_ESDR709(RGB_hdr_linear, dim=dim)

    return ERGB_sdr

def ICTCP_to_ICH(ITP, dim=-1):
    I, T, P = torch.split(ITP, 1, dim=dim)
    C = (T ** 2 + P ** 2) ** (1 / 2)
    H = torch.atan2(P,T)

    C = C * 2  #
    H = (H + 3.2) / 6.4  #

    ICH = torch.cat([I, C, H], dim=dim)
    return ICH


def ICH_to_ICTCP(ICH, dim=-1):
    I, C, H = torch.split(ICH, 1, dim=dim)
    C = C/2
    H = H*6.4-3.2

    T = C * torch.cos(H)
    P = C * torch.sin(H)
    ITP = torch.cat([I, T, P], dim=dim)
    return ITP


def test():
    sdrima = r'K:\HDRTVNet_data\test_set\test_sdr\036.png'
    hdrima = r'K:\HDRTVNet_data\test_set\test_hdr\036.png'

    ERGB_sdr = cv2.imread(sdrima, flags=-1)[:, :, ::-1] / 255
    ERGB_hdr = cv2.imread(hdrima, flags=-1)[:, :, ::-1] / 65535

    ITP_sdr = SDR_to_ICTCP(torch.from_numpy(ERGB_sdr))
    ITP_hdr = HDR_to_ICTCP(torch.from_numpy(ERGB_hdr))

    I_sdr, T_sdr, P_sdr = torch.split(ITP_sdr, 1, dim=2)
    H_sdr = torch.atan2(P_sdr,T_sdr)
    C_sdr = (T_sdr ** 2 + P_sdr ** 2) ** (1 / 2)

    I_hdr, T_hdr, P_hdr = torch.split(ITP_hdr, 1, dim=2)
    H_hdr = torch.atan2(P_hdr,T_hdr)
    C_hdr = (T_hdr ** 2 + P_hdr ** 2) ** (1 / 2)

    H_sdr = (H_sdr + 3.2) / 6.4
    H_hdr = (H_hdr + 3.2) / 6.4
    C_sdr = C_sdr * 2
    C_hdr = C_hdr * 2

    print(I_sdr.max(), I_sdr.min())
    print(H_sdr.max(), H_sdr.min())
    print(C_sdr.max(), C_sdr.min())
    print()
    print(I_hdr.max(), I_hdr.min())
    print(H_hdr.max(), H_hdr.min())
    print(C_hdr.max(), C_hdr.min())
