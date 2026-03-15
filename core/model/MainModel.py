import torch
from torch import nn
import dgl
from config.exp_config import Config
from core.model.Classifier import Classifier
from core.model.Voter import Voter
from core.model.Encoder import Encoder


class MainModel(nn.Module):
    def __init__(self, config: Config):
        super(MainModel, self).__init__()
        self.modalities = list(config.modalities)
        self.use_simple_gating = getattr(config, "use_simple_gating", False)
        self.simple_gating_mode = getattr(config, "simple_gating_mode", "concat")
        if self.simple_gating_mode not in {"sum", "concat", "mlp_concat"}:
            raise ValueError(f"Unsupported simple_gating_mode: {self.simple_gating_mode}")
        self._last_gate_info = None

        self.encoders = nn.ModuleDict()
        for modality in self.modalities:
            self.encoders[modality] = Encoder(
                alert_embedding_dim=config.alert_embedding_dim,
                graph_hidden_dim=config.graph_hidden_dim,
                graph_out_dim=config.graph_out,
                num_layers=config.graph_layers,
                aggregator=config.aggregator,
                feat_drop=config.feat_drop
            )

        if self.use_simple_gating:
            num_modalities = len(self.modalities)
            self.fti_gate_logits = nn.Parameter(torch.zeros(num_modalities))
            self.rcl_gate_logits = nn.Parameter(torch.zeros(num_modalities))
            if self.simple_gating_mode == "mlp_concat":
                in_dim = num_modalities * config.graph_out
                hidden_dim = max(in_dim // 2, num_modalities)
                self.fti_gate_net = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, num_modalities),
                )
                self.rcl_gate_net = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, num_modalities),
                )
        if self.use_simple_gating and self.simple_gating_mode == "sum":
            fti_fuse_dim = config.graph_out
            rcl_fuse_dim = config.graph_out
        else:
            fti_fuse_dim = len(self.modalities) * config.graph_out
            rcl_fuse_dim = len(self.modalities) * config.graph_out

        self.locator = Voter(rcl_fuse_dim,
                             hiddens=config.linear_hidden,
                             out_dim=1)
        self.typeClassifier = Classifier(in_dim=fti_fuse_dim,
                                         hiddens=config.linear_hidden,
                                         out_dim=config.ft_num)

    def _encode_modalities(self, batch_graphs):
        fs, es = {}, {}
        for modality in self.modalities:
            encoder = self.encoders[modality]
            x_d = batch_graphs.ndata[modality]
            f_d, e_d = encoder(batch_graphs, x_d) # graph-level, node-level
            fs[modality] = f_d
            es[modality] = e_d
        return fs, es

    def _fuse_modalities(self, fs, es, batch_graphs):
        if not self.use_simple_gating:
            return (
                torch.cat([fs[modality] for modality in self.modalities], dim=1),
                torch.cat([es[modality] for modality in self.modalities], dim=1),
            )

        if self.simple_gating_mode == "sum":
            fti_gate = torch.softmax(self.fti_gate_logits, dim=0)
            rcl_gate = torch.softmax(self.rcl_gate_logits, dim=0)
            f = torch.stack(
                [fs[modality] * fti_gate[idx] for idx, modality in enumerate(self.modalities)],
                dim=1
            ).sum(dim=1)
            e = torch.stack(
                [es[modality] * rcl_gate[idx] for idx, modality in enumerate(self.modalities)],
                dim=1
            ).sum(dim=1)
            self._last_gate_info = {
                "fti": fti_gate.detach(),
                "rcl": rcl_gate.detach(),
                "modalities": self.modalities,
                "mode": self.simple_gating_mode
            }
            return f, e

        if self.simple_gating_mode == "concat":
            fti_gate = torch.softmax(self.fti_gate_logits, dim=0)
            rcl_gate = torch.softmax(self.rcl_gate_logits, dim=0)
            f = torch.cat(
                [fs[modality] * fti_gate[idx] for idx, modality in enumerate(self.modalities)],
                dim=1
            )
            e = torch.cat(
                [es[modality] * rcl_gate[idx] for idx, modality in enumerate(self.modalities)],
                dim=1
            )
            self._last_gate_info = {
                "fti": fti_gate.detach(),
                "rcl": rcl_gate.detach(),
                "modalities": self.modalities,
                "mode": self.simple_gating_mode
            }
            return f, e

        f_graph = torch.cat([fs[modality] for modality in self.modalities], dim=1)
        fti_gate = torch.softmax(self.fti_gate_net(f_graph) + self.fti_gate_logits.view(1, -1), dim=1)
        rcl_graph_gate = torch.softmax(self.rcl_gate_net(f_graph) + self.rcl_gate_logits.view(1, -1), dim=1)
        num_nodes = batch_graphs.batch_num_nodes().to(rcl_graph_gate.device)
        rcl_gate = torch.repeat_interleave(rcl_graph_gate, num_nodes, dim=0)
        f = torch.cat(
            [fs[modality] * fti_gate[:, idx:idx + 1] for idx, modality in enumerate(self.modalities)],
            dim=1
        )
        e = torch.cat(
            [es[modality] * rcl_gate[:, idx:idx + 1] for idx, modality in enumerate(self.modalities)],
            dim=1
        )
        self._last_gate_info = {
            "fti": fti_gate.detach().mean(dim=0),
            "rcl": rcl_graph_gate.detach().mean(dim=0),
            "modalities": self.modalities,
            "mode": self.simple_gating_mode
        }
        return f, e

    def forward(self, batch_graphs):
        fs, es = self._encode_modalities(batch_graphs)
        f, e = self._fuse_modalities(fs, es, batch_graphs)

        # failure type identification
        type_logit = self.typeClassifier(f)

        # root cause localization
        root_logit = self.locator(e)

        return fs, es, root_logit, type_logit


    def message_aggregator(self, batch_graphs):
        fs, es = self._encode_modalities(batch_graphs)
        f, e = self._fuse_modalities(fs, es, batch_graphs)
        return f, e

    def get_gate_weights(self):
        if not self.use_simple_gating:
            return {}
        if self._last_gate_info is not None:
            return self._last_gate_info
        return {}
