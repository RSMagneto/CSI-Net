import torch
import torch.nn as nn
from models.encoder2 import build_backbone,SpatialAttention, ChannelAttention, PixelAttention,ChannelAttention2
from thop import profile
from thop import clever_format
# from cga import SpatialAttention, ChannelAttention, PixelAttention

def calc_men_std(feat,eps =1e-5):
    size = feat.size()
    assert (len(size)==4)
    N,C,H,W=size[:4]
    feat_var = feat.var(dim=1)+eps
    feat_var =feat_var.sqrt().view(N,1,H,W)
    feat_mean = feat.mean(dim=1).view(N,1,H,W)
    return feat_mean,feat_var

class cloor(nn.Module):
    def __init__(self):
        super(cloor, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1,bias=False),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1,bias=False),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
    def forward(self, T1, T2):
        T1_mean, T1_var = calc_men_std(T1)
        T2_mean, T2_var = calc_men_std(T2)
        # print(T2_mean.shape)
        # print(T2_var.shape)
        T = torch.cat((T1, T2), dim=1)
        x = self.conv1(T)
        # print(x.shape)
        T_mean = T1_mean-T2_mean
        T_var = T1_var-T2_var
        x_mean = x * T_mean
        # print(x_mean.shape)
        x_var = x * T_var
        X = torch.cat((x_mean,x_var), dim=1)
        X=self.conv2(X)
        return X



class GRAPHLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(GRAPHLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class graph(nn.Module):
    def __init__(self,bathsize):
        super(graph, self).__init__()
        self.bathsize = bathsize
        n1 = 32  # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]  # 0-32,1-64,2-128,3-256,4-512
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=filters[4]*2, out_channels=filters[4]*2, kernel_size=3, padding=1,bias=False),
            nn.Conv2d(in_channels=filters[4]*2, out_channels=filters[4], kernel_size=1, bias=False),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU()
        )
        # self.LayerNorm = GRAPHLayerNorm(4, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
        self.reps_graph = nn.Sequential(
            GRAPHLayerNorm(filters[4], eps=1e-12),
            nn.Dropout(0.1),
            nn.Linear(filters[4], filters[4]),
            nn.ReLU(True),
            GRAPHLayerNorm(filters[4], eps=1e-12),
        )
        # self.LayerNorm1 = GRAPHLayerNorm(4, eps=1e-12)
        self.dropout1 = nn.Dropout(0.1)
        self.reps_graph1 = nn.Sequential(
            GRAPHLayerNorm(filters[4], eps=1e-12),
            nn.Dropout(0.1),
            nn.Linear(filters[4], filters[4]),
            nn.ReLU(True),
            GRAPHLayerNorm(filters[4], eps=1e-12),
        )
    def forward(self,T1,T2):
        X= torch.cat((T1,T2),dim=1)
        X=self.conv1(X)
        x = X.permute(0,2,3,1)
        x = x.reshape(self.bathsize,4096,512)#imagesize=256
        # x = x.reshape(self.bathsize,1024,512)#imagesize=128
        # x = x.reshape(self.bathsize,9216,512)#imagesize=384
        reps_graph = torch.matmul(x, x.permute(0, 2, 1))
        reps_graph = nn.Softmax(dim=-1)(reps_graph)
        reps_graph = self.dropout(reps_graph)
        rel_reps = torch.matmul(reps_graph, x)
        rel_reps = self.reps_graph(x + rel_reps)
        reps_graph1 = torch.matmul(rel_reps, rel_reps.permute(0, 2, 1))
        reps_graph1 = nn.Softmax(dim=-1)(reps_graph1)
        reps_graph1 = self.dropout1(reps_graph1)
        rel_reps1 = torch.matmul(reps_graph1, rel_reps)
        rel_reps1 = self.reps_graph(rel_reps + rel_reps1)
        rel_reps1 = rel_reps1.reshape(self.bathsize, 64, 64, 512)#imagesize=256
        # rel_reps1 = rel_reps1.reshape(self.bathsize,32,32,512)
        # rel_reps1 = rel_reps.reshape(self.bathsize,96,96,512)
        rel_reps1 = rel_reps1.permute(0, 3, 1, 2)
        return rel_reps1

class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8): #dim指的是输入tensor的通道数，该模块输入与输出相同
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, color_feat, graph_feat):
        initial = color_feat + graph_feat
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * color_feat + (1 - pattn2) * graph_feat
        result = self.conv(result)
        return result


class CGAFusion2(nn.Module):
    def __init__(self, dim, reduction=8): #dim指的是输入tensor的通道数，该模块输入与输出相同
        super(CGAFusion2, self).__init__()
        self.sa = SpatialAttention()
        self.convfusion = nn.Conv2d(1024, 512,3,padding=1)
        self.ca = ChannelAttention2(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,t1,t2, color_feat, graph_feat):
        normal_feat = torch.cat((t1,t2),dim=1)
        normal_feat = self.convfusion(normal_feat)

        initial = normal_feat + color_feat + graph_feat
        cattn,mattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn +mattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * color_feat + (1 - pattn2) * graph_feat
        result = self.conv(result)
        return result


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        n1 = 32  # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]#0-32,1-64,2-128,3-256,4-512
        self.deconv1 = nn.Sequential(
            nn.Conv2d(filters[4]*3, filters[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(),
            nn.Conv2d(filters[4], filters[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU()
        )
        self.up1 =nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv2 =  nn.Sequential(
            nn.Conv2d(filters[3]*3, filters[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(),
            nn.Conv2d(filters[3], filters[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU()
        )

        self.deconv3 = nn.Sequential(
            nn.Conv2d(filters[2] * 3, filters[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(),
            nn.Conv2d(filters[2], filters[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU()
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv4 = nn.Sequential(
            nn.Conv2d(filters[1] * 3, filters[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(),
            nn.Conv2d(filters[1], filters[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU()
        )

        self.finalconv =nn.Conv2d(filters[0],2,kernel_size=3, stride=1, padding=1)


    def forward(self,T1,T2,feat_out,feat4_1,feat4_2,feat3_1,feat3_2,feat2_1,feat2_2):#f2是64*256*256 f3是128*128*128 f4是256*64*64 x1是512*64*64
        T = torch.cat((T1, T2,feat_out), dim=1)
        x4 = self.deconv1(T)
        x3 = self.deconv2(torch.cat((x4,feat4_1,feat4_2),dim=1))
        x3 = self.up1(x3)#64*64->128*128
        # print(x3.shape)
        # print(feat3_1.shape)
        # print(feat3_2.shape)
        x2 = self.deconv3(torch.cat((x3,feat3_1,feat3_2),dim=1))
        x2 = self.up2(x2)#128*128->256*256
        x1 = self.deconv4(torch.cat((x2,feat2_1,feat2_2),dim=1))
        result = self.finalconv(x1)

        return result

# class CNN(nn.Module):
#     def __init__(self,  backbone='resnet18', output_stride=16, f_c=64, freeze_bn=False, in_c=3):
#         super(CNN, self).__init__()
#         BatchNorm = nn.BatchNorm2d
#         self.backbone = build_backbone(backbone, output_stride, BatchNorm, in_c)
#
#     def forward(self,img):
#         x_1, f2, f3, f4 = self.backbone(img)#f2是64*256*256 f3是128*128*128 f4是256*64*64 x1是512*64*64
#         return x_1,f2, f3, f4


class GNETv3(nn.Module):
    def __init__(self,batchsize,backbone='resnet18', output_stride=16, f_c=64, freeze_bn=False, in_c=3):
        super(GNETv3, self).__init__()
        BatchNorm = nn.BatchNorm2d

        self.backbone=build_backbone(backbone, output_stride, BatchNorm, in_c)
        self.color = cloor()
        self.grahpconv = graph(batchsize)
        self.cgafusin=CGAFusion2(dim=512)
        self.decoder=decoder()

    def forward(self,T1,T2):
        x_1, f2_1, f3_1, f4_1 = self.backbone(T1)
        x_2, f2_2, f3_2, f4_2 = self.backbone(T2)
        feat_color = self.color(x_1,x_2)
        feat_graph =self.grahpconv(x_1,x_2)
        feat_fusion = self.cgafusin(x_1,x_2, feat_color,feat_graph)
        out = self.decoder(x_1,x_2,feat_fusion,f4_1,f4_2,f3_1,f3_2,f2_1,f2_2)
        output = []
        output.append(out)
        return output





# if __name__ == '__main__':
#     a = torch.randn(4, 3, 256, 256)
#     b= torch.randn(4, 3, 256, 256)
#     net = GNETv2(batchsize=4)
#     d = net(a, a)
#     flops,params = profile(net,inputs=(a,b))
#     flops,params=clever_format([flops,params],"%.3f")
#     print(params)
#     print(flops)

#     # print(c.shape)
#     # print(d.shape)
#     # print(e.shape)