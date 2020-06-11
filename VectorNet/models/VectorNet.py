import torch
import torch.nn as nn
import torch.nn.functional as F
import time
#torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_tensor_type(torch.FloatTensor)
#MLP
class MLP(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(MLP,self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
    
    def forward(self,x):
        output = self.MLP(x)
        return output

import dgl
def gcn_reduce(nodes):
    #agg_feature = torch.max(nodes.mailbox['msg'], dim=1)
    return {'v_feature': torch.max(nodes.mailbox['msg'], dim=1)[0]} #rel agg

#GCN
class GCNLayer(nn.Module):
    def __init__(self,in_feats,out_feats):
        super(GCNLayer,self).__init__()
    
    def forward(self,g,inputs):
        gcn_message = dgl.function.copy_src('v_feature','msg') 
        g.ndata['v_feature'] = inputs
        g.send(g.edges(),gcn_message)
        g.recv(g.nodes(),gcn_reduce)
        v_feature = g.ndata.pop('v_feature')
        return torch.cat([inputs,v_feature],dim = 1)

#Subgraph
class SubNetwork(nn.Module):
    def __init__(self,in_feats,hidden_size,layernums):
        super(SubNetwork,self).__init__()
        self.encoder = []
        self.gcnlayer = []
        self.layernums = layernums
        input_size = in_feats
        for i in range(0,layernums):
            if i == 0:
                self.encoder.append(MLP(input_size,hidden_size))
            else:
                self.encoder.append(MLP(hidden_size*2,hidden_size))
            self.gcnlayer.append(GCNLayer(hidden_size,hidden_size*2))
    
    def forward(self,g,inputs):
        g.ndata['v_feature'] = inputs
        g_batch = g
        for i in range(self.layernums):
            g_list = dgl.unbatch(g_batch)
            #进入MLP需要unbatch
            for subg in g_list:
                v_feature = self.encoder[i](subg.ndata['v_feature'])
                subg.ndata['v_feature'] = v_feature
            #进入gcn可以batch回来
            g_batch = dgl.batch(g_list)
            v_feature = self.gcnlayer[i](g_batch,g_batch.ndata['v_feature'])
            g_batch.ndata['v_feature'] = v_feature
        g_list = dgl.unbatch(g_batch)
        v_feature = []
        for subg in g_list:
            v_feature.append(subg.ndata['v_feature'])
        return torch.stack(v_feature,0)

#self-attention GAT
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = None #don't define the g before
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h }

    def forward(self, g,h):
        self.g = g
        # equation (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

#VectorNet model
class VectorNet(nn.Module):
    '''
    调用此模型即可
    '''
    def __init__(self,in_dim,hidden_size,out_dim):
        super(VectorNet,self).__init__()
        self.subMapNetwork = SubNetwork(in_dim,hidden_size,3)
        self.subAgentNetwork = SubNetwork(in_dim,hidden_size,3)
        self.GlobalNetwork = GATLayer(hidden_size*2,hidden_size*2)
        self.MLP = nn.Linear(hidden_size*2,out_dim)
    
    def forward(self,agent,map_set,agent_feature,map_feature,map_mask):
        MapOutputs = []
        Globalfeature = torch.max(self.subAgentNetwork(agent,agent_feature), dim=1)[0].unsqueeze(0)
        max_mask = torch.max(torch.sum(map_mask,dim = 1),dim = 0)[0].int()
        nodeN = 1 + max_mask
        for i,graph in enumerate(map_set):
            if i >= max_mask:
                break
            Globalfeature = torch.cat((Globalfeature,torch.max(self.subMapNetwork(graph,map_feature[i]), dim=1)[0].unsqueeze(0)),0)
        globalg = []
        for i in range(Globalfeature.shape[1]):
            Globalfeature[:,i]
            globalgraph = dgl.DGLGraph()
            globalgraph.add_nodes(nodeN,{'v_feature':Globalfeature[:,i]})
            src = []
            dst = []
            for i in range(nodeN):
                for j in range(nodeN):
                    if i != j:
                        src.append(i)
                        dst.append(j)
            globalgraph.add_edges(src,dst)
            globalg.append(globalgraph)
        g = dgl.batch(globalg)
        global_feature = self.GlobalNetwork(g,g.ndata['v_feature'])
        g.ndata['v_feature'] = global_feature
        globalg = dgl.unbatch(g)
        v_feature = []
        for subg in globalg:
            v_feature.append(subg.ndata['v_feature'][0])
        return self.MLP(torch.stack(v_feature,0))
    
    def save(self,name=None):
        if name is None:
            prefix = 'checkpoints/' + 'VectorNet' + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(),name)
        return name
    
    def load(self,path):
        self.load_state_dict(torch.load(path))
