import json
import os
import clip.clip as clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .. import datasets


with open('project_config.json', 'r') as f:
    project_config = json.load(f)

SYN_DATA_LOCATION = project_config['SYN_DATA_LOCATION']
CL_DATA_LOCATION = project_config['CL_DATA_LOCATION']
MTIL_DATA_LOCATION = project_config['MTIL_DATA_LOCATION']

def batch(iterable, n=64):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_datasets_text(ds, args):
    texts = []
    for d in ds:
        ref_sentences_cls = getattr(datasets, d)
        ref_sentences = ref_sentences_cls(
            None,
            location=args.data_location,
            batch_size=args.batch_size,
        )
        ref_template = ref_sentences.template
        ref_texts = [ref_template(x) for x in ref_sentences.classnames]
        texts.extend(ref_texts)
    ret = clip.tokenize(texts).cuda()
    return ret

def merge_we(model_0, model_1, sma_count):
    for param_q, param_k in zip(model_0.parameters(), model_1.parameters()):
        param_k.data = (param_k.data * sma_count + param_q.data) / (1.0 + sma_count)
    return model_1

def wise_we(model_0, model_1, sma_count, model_n, alpha=0.95):
    for param_q, param_k, param_n in zip(model_0.parameters(), model_1.parameters(), model_n.parameters()):
        param_k.data = (
                        (param_k.data * sma_count + param_q.data) / (1.0 + sma_count)
                    ) * alpha + param_n.data * (1-alpha)
    return model_1

def moving_avg(model_0, model_1, alpha=0.999):
    for param_q, param_k in zip(model_0.parameters(), model_1.parameters()):
        param_q.data = param_q.data * alpha + param_k.data * (1 - alpha)

def l2_loss(model, model_ref):
    loss = 0.0
    for param_q, param_k in zip(model.parameters(), model_ref.parameters()):
        loss += F.mse_loss(param_q, param_k.detach(), reduction="sum")
    return loss

def ewc_loss(model, model_ref, precision_matrices):
    loss = 0.0
    for (name_q, param_q), (_, param_k) in zip(model.named_parameters(), model_ref.named_parameters()):
        _loss = precision_matrices[name_q] * (param_q - param_k.detach()) ** 2
        loss += _loss.sum()
    return loss

def virtual_vocab(length=10, n_class=1000):
    voc_len = len(clip._tokenizer.encoder)
    texts = torch.randint(0, voc_len, (n_class, length))
    start = torch.full((n_class, 1), clip._tokenizer.encoder["<start_of_text>"])
    end = torch.full((n_class, 1), clip._tokenizer.encoder["<end_of_text>"])
    zeros = torch.zeros((n_class, 75 - length), dtype=torch.long)

    texts = torch.cat([start, texts, end, zeros], dim=1)
    return texts

# def distillation(t, s, T=2, reduction="mean"):
#      p = F.softmax(t / T, dim=1)
#      loss = F.cross_entropy(s / T, p, reduction=reduction) * (T ** 2)
#      return loss
def distillation(t, s, T=2):
    p = F.softmax(t / T, dim=1)
    log_q = F.log_softmax(s / T, dim=1)
    l = F.kl_div(log_q, p, reduction='batchmean') * (T ** 2)
    return l

#Manifold Alignment
def cosine_feature_loss(student_features, teacher_features):
    student_features = F.normalize(student_features, dim=-1)
    teacher_features = F.normalize(teacher_features, dim=-1)

    loss = 1.0 - (student_features * teacher_features).sum(dim=-1).mean()
    return loss

# Relational Knowledge Distillation - RKD
def rkd_loss(student_features, teacher_features):
    with torch.no_grad():
        t_d = torch.cdist(teacher_features, teacher_features, p=2)
        t_d_mean = t_d.mean()
        t_d = t_d / (t_d_mean + 1e-8)

    s_d = torch.cdist(student_features, student_features, p=2)
    s_d_mean = s_d.mean()
    s_d = s_d / (s_d_mean + 1e-8)

    loss = F.smooth_l1_loss(s_d, t_d)
    return loss
def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0)) - 2 * prod
    res = res.clamp(min=eps)
    if not squared:
        res = res.sqrt()
    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res

def rkd_angle_loss(student, teacher):
    with torch.no_grad():
        td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
        norm_td = F.normalize(td, p=2, dim=2)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

    sd = (student.unsqueeze(0) - student.unsqueeze(1))
    norm_sd = F.normalize(sd, p=2, dim=2)
    s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

    loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
    return loss


def rkd_distance_loss(student, teacher):
    with torch.no_grad():
        t_d = pdist(teacher, squared=False)
        mean_td = t_d[t_d > 0].mean()
        t_d = t_d / mean_td # 相对距离归一化

    s_d = pdist(student, squared=False)
    mean_sd = s_d[s_d > 0].mean()
    s_d = s_d / mean_sd

    loss = F.smooth_l1_loss(s_d, t_d, reduction='mean')
    return loss

def relation_distillation_loss(student_features, teacher_features, dist_ratio=2.0, angle_ratio=1.0):
    loss_dist = rkd_distance_loss(student_features, teacher_features)
    loss_angle = rkd_angle_loss(student_features, teacher_features)
    return dist_ratio * loss_dist + angle_ratio * loss_angle
    
def kl_divergence(t, s, T=2, reduction="batchmean"):
    p = F.log_softmax(t / T, dim=1)
    q = F.softmax(s / T, dim=1)
    loss = F.kl_div(p, q, reduction=reduction) * (T ** 2)
    return loss

def paired_loss_new(old_pred, old_true):
    T = 2
    pred_soft = F.softmax(old_pred[:, : old_true.shape[0]] / T, dim=1)
    true_soft = F.softmax(old_true[:, : old_true.shape[0]] / T, dim=1)
    loss_old = true_soft.mul(-1 * torch.log(pred_soft))
    loss_old = loss_old.sum(1)
    loss_old = loss_old.mean() * T * T
    return loss_old

def prepare_ref_dataset(args, test_preprocess):
    if args.ref_dataset == "SyntheticDataset_A":
        ref_dataset_cls = getattr(datasets, "SyntheticDataset")
        ref_dataset = ref_dataset_cls(
                test_preprocess,
                location=os.path.join(SYN_DATA_LOCATION, 'synthetic_data_a/' + args.train_dataset +'_Syn/'),
                batch_size=args.batch_size,
                num_workers=8,
                image_nums=args.image_nums,
            )
        
        ref_texts = ref_dataset.all_prompts
    elif args.ref_dataset == "SyntheticDataset_B":
        ref_dataset_cls = getattr(datasets, "SyntheticDataset")
        ref_dataset = ref_dataset_cls(
                test_preprocess,
                location=SYN_DATA_LOCATION+'synthetic_data_b/' + args.train_dataset +'_Syn/',
                batch_size=args.batch_size,
                num_workers=8,
            )
        ref_texts = ref_dataset.all_prompts
    elif args.ref_dataset == "ImageNetSUB":
        dataset_names =  ["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers", "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"]
        seed_offset = dataset_names.index(args.train_dataset)
        ref_dataset_cls = getattr(datasets, "ImageNetSUB")
        ref_dataset = ref_dataset_cls(
                test_preprocess,
                location=CL_DATA_LOCATION,
                batch_size=args.batch_size,
                num=args.image_nums,
                random_seed=42+seed_offset,
                num_workers=8,
            )
        ref_template = ref_dataset.template
        ref_texts = [ref_template(x) for x in ref_dataset.classnames]
    elif args.ref_dataset == "ImageNet":
        # Use ImageNet as reference 
        ref_dataset_cls = getattr(datasets, "ImageNet")
        ref_dataset = ref_dataset_cls(
                test_preprocess,
                location=CL_DATA_LOCATION,
                batch_size=args.batch_size,
                num_workers=8,
            )
        
        ref_template = ref_dataset.template
        ref_texts = [ref_template(x) for x in ref_dataset.classnames]

    return ref_dataset, ref_texts





def attention_transfer_loss(student_attns, teacher_attns, layers_to_match='last'):
    """
    计算 Attention Transfer Loss
    Args:
        student_attns: list of tensors, 学生的注意力图列表
        teacher_attns: list of tensors, 老师的注意力图列表
        layers_to_match: 'all' 匹配所有层, 'last' 只匹配最后一层
    """
    loss = 0.0
    T = 4.0
    if layers_to_match == 'last':
        s_map = student_attns[-1]
        t_map = teacher_attns[-1].detach()


        eps = 1e-8
        t_map = t_map + eps
        s_map = s_map + eps

        t_map = t_map / t_map.sum(dim=-1, keepdim=True)
        s_map = s_map / s_map.sum(dim=-1, keepdim=True)

        s_map_t = s_map.pow(1/T)
        s_map_t = s_map_t / s_map_t.sum(dim=-1, keepdim=True)

        t_map_t = t_map.pow(1/T)
        t_map_t = t_map_t / t_map_t.sum(dim=-1, keepdim=True)

        log_s_map = torch.log(s_map_t)
        
        loss = F.kl_div(log_s_map, t_map_t, reduction='batchmean') * (T**2)
        
    return loss

def relation_distillation_loss(student_features, teacher_features):
    """
    Relational Knowledge Distillation
    """
    s_norm = F.normalize(student_features, p=2, dim=1)
    t_norm = F.normalize(teacher_features, p=2, dim=1)

    G_s = torch.mm(s_norm, s_norm.t())
    G_t = torch.mm(t_norm, t_norm.t())

    loss = F.mse_loss(G_s, G_t)

    return loss






#SCI-PD
# def scipd_loss(student_features, teacher_features, scale=0.1):
#     """
#     SCI-PD: 使用余弦距离替代 MSE，数值量级更合适
#     """
#     with torch.no_grad():
#         std = student_features.std(dim=-1, keepdim=True)
    
#     # 生成扰动
#     perturbation = torch.randn_like(student_features) * std * scale
#     perturbed_student_features = student_features + perturbation
    
#     # 使用 Cosine Embedding Loss (1 - cos_sim)
#     # 结果范围通常在 [0, 2] 之间，量级与 CE Loss 更匹配
#     # target=1 表示希望两个向量相似
#     loss = 1.0 - F.cosine_similarity(perturbed_student_features, teacher_features, dim=-1).mean()
    
#     return loss
def scipd_loss(student_features, teacher_features, args):
    """
    SCI-PD (Cosine Version)
    
    参数说明:
    - student_features: 当前模型的特征
    - teacher_features: 如果你想做蒸馏就传 Teacher，如果你想做自监督一致性，这里应该传 student_features.detach()
    """
    
    # 1. 准备工作
    scale = float(args.sci_scale)  # e.g., 0.1
    
    # 计算标准差用于控制噪声幅度 (保持你代码中的逻辑)
    with torch.no_grad():
        std = student_features.std(dim=-1, keepdim=True)
    
    # 2. 生成随机高斯噪声 (无梯度)
    # torch.randn_like 生成标准正态分布
    noise = torch.randn_like(student_features) * std * scale
    
    # 3. 施加扰动
    perturbed_student_features = student_features + noise
    
    target = teacher_features 
    
    # 计算余弦相似度: output 范围 [-1, 1]
    cos_sim = F.cosine_similarity(perturbed_student_features, target, dim=-1)
    
    # Loss = 1 - 相似度 (范围 0 到 2，越小越好)
    loss = 1.0 - cos_sim.mean()
    
    return float(args.sci_weight) * loss