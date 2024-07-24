import torch
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
from timm import create_model
import typing
import warnings
class Resnet18(nn.Module):
    def __init__(self,out_num):
        super(Resnet18,self).__init__()
        self.model = models.resnet18(pretrained = True)
        self.feature = nn.Sequential(*list(self.model.children())[:-1])
        in_num = self.model.fc.in_features
        self.fc = nn.Linear(in_num,out_num)
    def forward(self,x):
        feature = self.feature(x).squeeze()
        output = F.softmax(self.fc(feature))
        return feature,output
class Resnet50(nn.Module):
    def __init__(self,out_num):
        super(Resnet50,self).__init__()
        self.model = models.resnet50(pretrained = True)
        self.feature = nn.Sequential(*list(self.model.children())[:-1])
        in_num = self.model.fc.in_features
        self.fc = nn.Linear(in_num, out_num)

    def forward(self, x):
        feature = self.feature(x).squeeze()
        output = F.softmax(self.fc(feature))
        return feature, output


class Densenet121(nn.Module):
    def __init__(self,out_num):
        super(Densenet121,self).__init__()
        self.model = models.densenet121(pretrained = True)
        in_num = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.fc = nn.Linear(in_num,out_num)
    def forward(self,x):
        feature = self.model(x)
        output = F.softmax(self.fc(feature))
        return feature,output
class Basic_Block(nn.Module):
    def __init__(self,in_num,out_num):
        super(Basic_Block,self).__init__()
        self.ln = nn.Linear(in_num,out_num)
        self.LayerNorm = nn.LayerNorm(out_num)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.block = nn.Sequential(self.ln,self.LayerNorm,self.relu,self.dropout)
    def forward(self,x):
        return self.block(x)

class Residual_Block(nn.Module):
    def __init__(self,block):
        super(Residual_Block,self).__init__()
        self.block = block
    def forward(self,x):
        return x+self.block(x)



class Swin_Transforemr(torch.nn.Module):
    def __init__(self,model_size,pretrained_dataset,out_num):
        '''

        :param model_size: 使用在1k上预训练的模型还是在22k上预训练的模型 --》tiny or base
        :param out_num:
        '''
        super(Swin_Transforemr,self).__init__()
        if pretrained_dataset == '1k':
            if model_size == 'b':
                pretrained_cfg = create_model('swin_base_patch4_window7_224').default_cfg
                pretrained_cfg['file'] = './swin_base_patch4_window7_224.pth'
                self.model = create_model('swin_base_patch4_window7_224', pretrained=True,
                                          pretrained_cfg=pretrained_cfg)
            elif model_size == 't':
                pretrained_cfg = create_model('swin_tiny_patch4_window7_224').default_cfg
                pretrained_cfg['file'] = './swin_tiny_patch4_window7_224.pth'
                self.model = create_model('swin_tiny_patch4_window7_224', pretrained=True,
                                          pretrained_cfg=pretrained_cfg)
        elif pretrained_dataset == '22k':
            if model_size == 'b':
                pretrained_cfg = create_model('swin_base_patch4_window7_224.ms_in22k').default_cfg
                pretrained_cfg['file'] = './swin_base_patch4_window7_224_in22k.pth'
                self.model = create_model('swin_base_patch4_window7_224_in22k', pretrained=True, pretrained_cfg=pretrained_cfg)
        self.ln = nn.Linear(1000,out_num)

    def forward(self,image):
        feature = self.model(image)
        return feature,F.softmax(self.ln(feature))

class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入的每个patch的嵌入维度
                 num_heads,
                 qkv_bias=False,  # 生成qkv时需不需要bias参数
                 qk_scale=None,  # 计算attn时要scale的量
                 attn_drop_ratio=0.,  # attn随机drop
                 proj_drop_ratio=0.):  # 最后projection后随机drop一些
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x,attn



class SelfAttention(nn.Module):
    def __init__(self,dim,num_heads,attn_drop_ratio=0.,proj_drop_ratio=0.):
        '''
        Args:
            dim: resnet18为49（7X7）
            num_heads: 应该按照dim的数量进行调整，可被整除
            attn_drop_ratio:
            proj_drop_ratio:
        '''
        super(SelfAttention,self).__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim,3*dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
    def forward(self,x):
        B,N,C = x.shape #x传入之前应该resize为B X C X dim
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        attn = torch.matmul(q,k.transpose(-2,-1)) * self.scale
        attn = F.softmax(attn,dim=-1)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn,v).transpose(1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x,attn


class LayerNorm(nn.Module):
    def __init__(self,hidden_size,epsilon = 1e-12):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.epsilon = epsilon
    def forward(self,x):
        u = x.mean(-1,keepdim = True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.epsilon)
        return self.gamma * x + self.beta





class MLP(nn.Module):
    def __init__(self,hidden_size,output_num,dropout_rate = 0.):
        super(MLP,self).__init__()
        self.dense = nn.Linear(hidden_size,128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(128,output_num)
    def forward(self,x):
        feature = self.dropout(self.relu(self.dense(x)))
        return feature,F.softmax(self.classifier(feature))


class CNN_SelfAttention(nn.Module):
    def __init__(self,model_name,out_num):
        super(CNN_SelfAttention,self).__init__()
        if model_name == 'resnet18':
            cnn = models.resnet18(pretrained = True)
            self.cnn = nn.Sequential(*list(cnn.children())[:-2])
            dim = 49
            self.channel_selfattention = SelfAttention(dim,7,0.3,0.3)
            self.layernorm = LayerNorm(49)
            self.spatial_selfattention = SelfAttention(512,num_heads=8,attn_drop_ratio=0.3,proj_drop_ratio=0.3)
            self.layernorm2 = LayerNorm(512)
            self.mlp = MLP(512,out_num,0.3)
        elif  model_name == 'densenet121':
            cnn = models.densenet121(pretrained = True)
            self.cnn = nn.Sequential(*list(cnn.children())[:-1])
            dim = 49
            self.channel_selfattention = SelfAttention(dim, 7, 0.3, 0.3)
            self.layernorm = LayerNorm(49)
            self.spatial_selfattention = SelfAttention(1024,num_heads=8,attn_drop_ratio=0.3,proj_drop_ratio=0.3)
            self.layernorm2 = LayerNorm(1024)
            self.mlp = MLP(1024, out_num, 0.3)
        else:
            raise Exception('model is wrong! Only use resnet50 or densenet121')

    def forward(self,x,attn = False):
        x = self.cnn(x)
        B,C,w,h = x.shape
        x = x.reshape(B,C,w*h)
        residual_channel = x
        x_channel,attn_channel = self.channel_selfattention(x)
        x_channel += residual_channel
        x_channel = self.layernorm(x_channel)

        residual_spatial = x_channel.reshape(B,x_channel.shape[2],C)
        x_spatial = residual_spatial
        x_spatial,attn_spatial = self.spatial_selfattention(x_spatial)
        x_spatial += residual_spatial
        x_spatial = self.layernorm2(x_spatial)
        x = x_spatial.permute(0,2,1)

        maxpooling = nn.MaxPool1d(x.shape[2])
        x = maxpooling(x).squeeze()
        feature,output = self.mlp(x)
        if attn:
            return feature,output,attn_channel,attn_spatial
        else:
            return feature,output
#===========定义cnn-sasm================
class CNN_SelfAttention_softmax(nn.Module):
    def __init__(self,model_name,out_num):
        super(CNN_SelfAttention_softmax,self).__init__()
        if model_name == 'resnet18':
            cnn = models.resnet18(pretrained = True)
            self.cnn = nn.Sequential(*list(cnn.children())[:-2])
            dim = 49
            self.channel_selfattention = SelfAttention(dim,7,0.3,0.3)
            self.layernorm = LayerNorm(49)
            self.spatial_selfattention = SelfAttention(512,num_heads=8,attn_drop_ratio=0.3,proj_drop_ratio=0.3)
            self.layernorm2 = LayerNorm(512)
            self.mlp = MLP(512,out_num,0.3)
        elif  model_name == 'densenet121':
            cnn = models.densenet121(pretrained = True)
            self.cnn = nn.Sequential(*list(cnn.children())[:-1])
            dim = 49
            self.channel_selfattention = SelfAttention(dim, 7, 0.3, 0.3)
            self.layernorm = LayerNorm(49)
            self.spatial_selfattention = SelfAttention(1024,num_heads=8,attn_drop_ratio=0.3,proj_drop_ratio=0.3)
            self.layernorm2 = LayerNorm(1024)
            self.mlp = MLP(1024, out_num, 0.3)
        else:
            raise Exception('model is wrong! Only use resnet50 or densenet121')

    def forward(self,x,attn = False):
        x = self.cnn(x)
        B,C,w,h = x.shape
        x = x.reshape(B,C,w*h)
        residual_channel = x
        x_channel,attn_channel = self.channel_selfattention(x)
        x_channel += residual_channel
        x_channel = F.softmax(self.layernorm(x_channel),dim=-2)

        residual_spatial = x_channel.reshape(B,x_channel.shape[2],C)
        x_spatial = residual_spatial
        x_spatial,attn_spatial = self.spatial_selfattention(x_spatial)
        x_spatial += residual_spatial
        x_spatial = F.softmax(self.layernorm2(x_spatial),dim=-2)
        x = x_spatial.permute(0,2,1)

        maxpooling = nn.MaxPool1d(x.shape[2])
        x = maxpooling(x).squeeze()
        feature,output = self.mlp(x)
        if attn:
            return feature,output,attn_channel,attn_spatial
        else:
            return feature,output



class Channel_Attention(nn.Module):
    def __init__(self,in_planes,ratio = 16):
        super(Channel_Attention,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes,in_planes//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes//ratio,in_planes,1,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CNN_SelfAttention_attn(nn.Module):
    def __init__(self,model_name,out_num):
        super(CNN_SelfAttention_attn,self).__init__()
        if model_name == 'resnet18':
            cnn = models.resnet18(pretrained = True)
            self.cnn = nn.Sequential(*list(cnn.children())[:-2])
            dim = 49
            self.channel_selfattention = SelfAttention(dim,7,0.3,0.3)
            self.layernorm = LayerNorm(49)
            self.channel_attn = Channel_Attention(512)
            self.spatial_selfattention = SelfAttention(512,num_heads=8,attn_drop_ratio=0.3,proj_drop_ratio=0.3)
            self.layernorm2 = LayerNorm(512)
            self.spatial_attn = SpatialAttention()
            self.mlp = MLP(512,out_num,0.3)
        elif  model_name == 'densenet121':
            cnn = models.densenet121(pretrained = True)
            self.cnn = nn.Sequential(*list(cnn.children())[:-1])
            dim = 49
            self.channel_selfattention = SelfAttention(dim, 7, 0.3, 0.3)
            self.layernorm = LayerNorm(49)
            self.channel_attn = Channel_Attention(1024)
            self.spatial_selfattention = SelfAttention(1024,num_heads=8,attn_drop_ratio=0.3,proj_drop_ratio=0.3)
            self.layernorm2 = LayerNorm(1024)
            self.spatial_attn = SpatialAttention()
            self.mlp = MLP(1024, out_num, 0.3)
        else:
            raise Exception('model is wrong! Only use resnet50 or densenet121')

    def forward(self,x,attn = False):
        x = self.cnn(x)
        B,C,w,h = x.shape
        x = x.reshape(B,C,w*h)
        residual_channel = x
        x_channel,attn_channel = self.channel_selfattention(x)
        x_channel += residual_channel
        x_channel = self.layernorm(x_channel).reshape(B,C,w,h)
        x_channel = self.channel_attn(x_channel) * x_channel

        residual_spatial = x_channel.reshape(B,w*h,C)
        x_spatial = residual_spatial
        x_spatial,attn_spatial = self.spatial_selfattention(x_spatial)
        x_spatial += residual_spatial
        x_spatial = self.layernorm2(x_spatial)
        x = x_spatial.permute(0,2,1).reshape(B,C,w,h)
        x = self.spatial_attn(x) * x

        maxpooling = nn.AdaptiveMaxPool2d(1)
        x = maxpooling(x).squeeze()
        feature,output = self.mlp(x)
        if attn:
            return feature,output,attn_channel,attn_spatial
        else:
            return feature,output


class Simple_Concat_oneModality(nn.Module):
    def __init__(self,in_dim):
        super(Simple_Concat_oneModality,self).__init__()
        self.patho_ln = Basic_Block(in_dim,128)
        self.clincal_ln = Basic_Block(7,128)
        self.therapy_ln = Basic_Block(128,128)

        self.fn = nn.Linear(128*2,128)
        self.layernorm = nn.LayerNorm(128)
        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(128,2)
    def forward(self,patho_fea=None,clinical_fea=None,therapy_fea=None):
        therapy_fea = therapy_fea.repeat(1, 128)
        therapy_fea = self.therapy_ln(therapy_fea.to(torch.float32))  # 128->128
        if patho_fea==None:
            clinical_fea = self.clincal_ln(clinical_fea.to(torch.float32)) # 7->128
            # therapy_fea = therapy_fea.unsqueeze(-1)
            x = torch.concat((clinical_fea, therapy_fea), dim=-1)
            x = self.dropout(F.relu(self.layernorm(self.fn(x))))
            output = F.softmax(self.classifier(x))
        elif clinical_fea==None:
            patho_fea = self.patho_ln(patho_fea.to(torch.float32)) #512->128
            # clinical_fea = self.clincal_ln(clinical_fea.to(torch.float32)) # 7->128
            # therapy_fea = therapy_fea.unsqueeze(-1)
            x = torch.concat((patho_fea,therapy_fea),dim=-1)
            x = self.dropout(F.relu(self.layernorm(self.fn(x))))
            output = F.softmax(self.classifier(x))
        return output



#=============仅用病理特征预测复发===============#
class Pathological_predict_prognosis(nn.Module):
    def __init__(self,in_dim):
        super(Pathological_predict_prognosis,self).__init__()
        self.dense = nn.Linear(in_dim,256)
        self.bn = nn.BatchNorm1d(256)
        self.drop = nn.Dropout()

        self.dense2 = nn.Linear(256,in_dim)
        self.bn2 = nn.BatchNorm1d(in_dim)
        self.drop2 = nn.Dropout()

        self.classifier = nn.Linear(in_dim,2)

    def forward(self,pathological_feature):
        x = self.drop(F.relu(self.bn(self.dense(pathological_feature))))
        x = self.drop2(F.relu(self.bn2(self.dense2(x))))
        return F.softmax(self.classifier(x))






if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # net = CNN_SelfAttention_attn('resnet18',2)
    # # net = SpatialAttention()
    # a = torch.randn(2,3,224,224)
    # b,c = net(a)
    # print(b.shape)
    # print(c)

    # config = [[32,32,32],[64,64,64],[128,128,256]]
    # net = Fusion_model(128,config,2)
    # total_trainable_params = sum(
    #     p.numel() for p in net.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    # print(f'{total_trainable_params*4/(1024*1024):.2f}M training parameters.')


