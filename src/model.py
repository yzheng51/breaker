import torch


class UserRepresentation(torch.nn.Module):
    def __init__(self, n_feat, n_field, embed_size=5):
        super().__init__()
        self.user_embed = torch.nn.Embedding(n_feat, embed_size)
        self.user_rep = torch.nn.Sequential(
            torch.nn.Linear(n_field * embed_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
        )

        self.init_weights()

    def forward(self, u_feat):
        user_embed = self.user_embed(u_feat).view(u_feat.size(0), -1)
        user_embed = self.user_rep(user_embed)

        return user_embed

    def init_weights(self):
        torch.nn.init.normal_(self.user_embed.weight, mean=0, std=0.01)


class Breaker(torch.nn.Module):
    def __init__(self, n_feat, n_field, n_clusters, init_user_embed, init_cluster_centers, embed_size=5, alpha=1):
        super().__init__()
        self.alpha = alpha
        self.user_embed = UserRepresentation(n_feat, n_field, embed_size=embed_size)
        self.user_embed.load_state_dict(init_user_embed.state_dict())
        self.item_embed = torch.nn.Embedding(10, embed_size)
        self.tower_ls = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(64 + embed_size, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 1),
            ) for _ in range(n_clusters)
        ])
        self.cluster_centers = torch.nn.Parameter(torch.from_numpy(init_cluster_centers))

        self.init_weights()

    def forward(self, u_feat, i_feat):
        item_embed = self.item_embed(i_feat).view(i_feat.size(0), -1)
        user_embed = self.user_embed(u_feat)

        tower_output = [module(torch.cat([user_embed, item_embed], 1)) for module in self.tower_ls]
        tower_output = torch.sigmoid(torch.cat(tower_output, 1))

        q = 1 / (1 + (user_embed.unsqueeze(1) - self.cluster_centers).square().sum(2) / self.alpha) ** ((self.alpha + 1) / 2)
        # q = q ** ((self.alpha + 1) / 2)
        q = q / q.sum(1, keepdim=True)

        return q, (tower_output * q).sum(1)

    def init_weights(self):
        torch.nn.init.normal_(self.item_embed.weight, mean=0, std=0.01)
