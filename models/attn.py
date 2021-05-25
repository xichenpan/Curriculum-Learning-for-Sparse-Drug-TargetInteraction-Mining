# 05:03 5/24/2021 Haotian Xue

import torch.nn as nn
import torch


class Attn_module(nn.Module):
    def __init__(self, embed_dim, kdim, vdim, proj_bias=True):
        super(Attn_module, self).__init__()

        # QKV projection layer
        self.k_proj = nn.Linear(kdim, embed_dim, bias=proj_bias)
        self.v_proj = nn.Linear(vdim, embed_dim, bias=proj_bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=proj_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=proj_bias)

    def forward(self, q, k, v, key_padding_mask, query_padding_mask):
        """
        attention from q -> (k, v)
        :param q:  bs * q_len * q_dim
        :param k: bs * k_len * k_dim
        :param v: bs * v_len * v_dim
        :param key_padding_mask: bs * k_len [0,0,0,0,1,1,1] e.g.
        :param query_padding_mask: bs * q_len
        :return: bs * q_len * q_dim
        """

        assert k.shape[1] == v.shape[1]

        # --- proj QKV ---
        # q : bs * q_len * emd
        # k : bs * k_len * emd
        # v : bs * v_len * emd
        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)

        # --- get attention weights ---

        attn_weights = torch.bmm(q, k.transpose(1, 2))  # bs * q_len * k_len
        attn_weights = attn_weights.transpose(1, 2)  # bs * k_len * q_len
        key_padding_mask = key_padding_mask[:, :, [0]]

        attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))

        attn_weights = attn_weights.transpose(1, 2)  # bs * q_len * k_len
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)  # bs * q_len * k_len

        # --- get attention and out projection ---
        attn = torch.bmm(attn_weights, v)
        attn = self.out_proj(attn)

        return attn, attn_weights


class Drug_Target_Cross_Attnention(nn.Module):
    def __init__(self, drug_feature_dim, target_feature_dim, layer_num, proj_bias=True):
        super(Drug_Target_Cross_Attnention, self).__init__()

        self.drug_to_target_attn_layers = nn.ModuleList([])
        self.target_to_drug_attn_layers = nn.ModuleList([])
        self.layer_num = layer_num

        for i in range(layer_num):
            self.drug_to_target_attn_layers.append(
                Attn_module(
                    embed_dim=drug_feature_dim,
                    kdim=target_feature_dim,
                    vdim=target_feature_dim,
                    proj_bias=proj_bias
                )
            )

            self.target_to_drug_attn_layers.append(
                Attn_module(
                    embed_dim=target_feature_dim,
                    kdim=drug_feature_dim,
                    vdim=drug_feature_dim,
                    proj_bias=proj_bias
                )
            )

    def forward(self, drug_input, target_input, drug_mask, target_mask):
        """

        :param drug_input: bs * len_drug * drug_dim
        :param target_input: bs * len_tar * tar_dim
        :param drug_mask: bs * len_drug
        :param target_mask: bs * len_tar
        :return: {"attn_drug":, "attn_target":}
        """
        drug_state = drug_input
        target_state = target_input
        for layer_id in range(self.layer_num):
            drug_state_new, _ = self.drug_to_target_attn_layers[layer_id](
                q=drug_state,
                k=target_state,
                v=target_state,
                key_padding_mask=target_mask,
                query_padding_mask=drug_mask
            )
            target_state_new, _ = self.target_to_drug_attn_layers[layer_id](
                q=target_state,
                k=drug_state,
                v=drug_state,
                key_padding_mask=drug_mask,
                query_padding_mask=target_mask
            )
            drug_state = drug_state_new
            target_state = target_state_new

        return {
            "attn_drug": drug_state,  # bs * len_drug * drug_dim
            "attn_target": target_state  # bs * len_tar * tar_dim
        }


class Drug_Target_Cross_Attnention_Pooling(nn.Module):
    def __init__(self, drug_feature_dim, target_feature_dim, layer_num, proj_bias=True):
        super(Drug_Target_Cross_Attnention_Pooling, self).__init__()
        self.drug_to_target_attn_layer = Attn_module(
            embed_dim=drug_feature_dim,
            kdim=target_feature_dim,
            vdim=target_feature_dim,
            proj_bias=proj_bias
        )

        self.target_to_drug_attn_layer = Attn_module(
            embed_dim=target_feature_dim,
            kdim=drug_feature_dim,
            vdim=drug_feature_dim,
            proj_bias=proj_bias
        )

    def forward(self, drug_input, target_input, drug_mask, target_mask):
        """

        :param drug_input:
        :param target_input:
        :param drug_mask:
        :param targe_mask:
        :return:
        """

        drug_k = drug_input.mean(dim=-2, keepdim=True)
        target_k = target_input.mean(dim=-2, keepdim=True)

        target_rep, _ = self.drug_to_target_attn_layer(
            q=drug_k,
            k=target_input,
            v=target_input,
            key_padding_mask=target_mask,
            query_padding_mask=drug_mask
        )

        drug_rep, _ = self.target_to_drug_attn_layer(
            q=target_k,
            k=drug_input,
            v=drug_input,
            key_padding_mask=drug_mask,
            query_padding_mask=target_mask
        )

        return drug_rep, target_rep


if __name__ == "__main__":
    device = 1

    drug = torch.rand(3, 100, 512)
    target = torch.rand(3, 3000, 600)

    drug_padding_mask = ~ (torch.zeros(3, 100) == 0)
    target_padding_mask = ~ (torch.zeros(3, 3000) == 0)

    drug_padding_mask[:, -20:] = True
    target_padding_mask[:, -100] = True

    # model = Drug_Target_Cross_Attnention(drug_feature_dim=512, target_feature_dim=600, layer_num=3, proj_bias=True)
    # print(model)
    #
    # model = model.to(device)
    # drug, target = drug.to(device), target.to(device)
    # drug_padding_mask, target_padding_mask = drug_padding_mask.to(device), target_padding_mask.to(device)
    #
    # res = model(drug, target, drug_padding_mask, target_padding_mask)
    #
    # print(res["attn_drug"].shape, res["attn_target"].shape)

    model = Drug_Target_Cross_Attnention_Pooling(drug_feature_dim=512, target_feature_dim=600, layer_num=3, proj_bias=True)
    print(model)

    model = model.to(device)
    drug, target = drug.to(device), target.to(device)
    drug_padding_mask, target_padding_mask = drug_padding_mask.to(device), target_padding_mask.to(device)

    res = model(drug, target, drug_padding_mask, target_padding_mask)

    print(res[0].shape, res[1].shape)
